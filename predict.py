import torch
import numpy as np
from predictmdl.models.pointnet2_cls_ssg import get_model
import predictmdl.utils.pointcloud_utils as pcu
from pyntcloud import PyntCloud
import os
import pandas as pd
from tqdm import tqdm
import settings.seg_settings as ss 

def farthest_point_sample(xyz, npoint):
    device = xyz.device
    batchsize, ndataset, dimension = xyz.shape
    centroids = torch.zeros(batchsize, npoint, dtype=torch.long).to(device)
    distance = torch.ones(batchsize, ndataset).to(device) * 1e10
    farthest =  torch.randint(0, ndataset, (batchsize,), dtype=torch.long).to(device)
    batch_indices = torch.arange(batchsize, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:,i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batchsize, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def test(src, model_name):

    model_path = 'predictmdl/checkpoints/'+ model_name +'/models/model.t7'

    species_names = ['Tree','Not_Tree']
    # species_names = ['E','C','B','R','D','OC']
    # species_names = ['E','C','B']
    try:
        pc = PyntCloud.from_file(src)
        points = pc.points.loc[:,["x","y","z"]].values
        points = np.array([points])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        points = torch.Tensor(points).to(device)
        centroids = farthest_point_sample(points, 2048)
        pc_sampled = points[0][centroids[0]]
        pc_sampled = pc_sampled.cpu().detach().numpy()

        X_test = np.array([pc_sampled])
        y_test = [0]

        X_test = pcu.tree_normalize(X_test)
        int2name = { i:name for i, name in enumerate(species_names)}

        NUM_CLASSES = len(int2name)

        model = get_model(NUM_CLASSES,normal_channel=False).to(device)
        model.load_state_dict(torch.load(model_path))
        model = model.eval()
        test_true = []
        test_pred = []
        data, label = torch.tensor(X_test, device=device), torch.tensor(y_test, device=device)
        data = data.permute(0, 2, 1)
        logits, trans_feat = model(data)
        preds = logits.max(dim=1)[1].detach()
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.cpu().numpy())
        if test_pred[0][0] == 1:
            ans = 0 #"Это не дерево"
        else:
            ans = 1 #"Это дерево"
    except:
        ans = -1
    return ans
    # return test_pred[0][0]

def predict(path_file, model_name):
    names = []
    labels = []
    for filename in tqdm(os.listdir(path_file)):
        if filename.endswith('.pcd'):
            src = os.path.join(path_file,filename)
            label = test(src, model_name)
            names.append(filename)
            labels.append(label)     
    bd = pd.DataFrame({"Name_tree": names,"Label": labels})
    bd.to_csv(os.path.join(path_file,'predict_' + model_name + '.csv'), index = False, sep=';')

if __name__ == '__main__':
    model_name = 'int0000_7000-512-rlish-s4762'
    path_file = os.path.join(path_file, 'int0000_7000-512-rlish-s4762')
    predict(path_file, model_name)     
    