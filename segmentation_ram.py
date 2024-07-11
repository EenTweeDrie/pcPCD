import pandas as pd
import os
from scipy.spatial.distance import cdist
from settings.seg_settings import SS
from classes.RAM import RAM
import numpy as np

def makedirs_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def segmentation_ram(ss):

    path_file_save = os.path.join(ss.path_base, ss.step1_folder_name, ss.step2_folder_name)
    makedirs_if_not_exist(path_file_save)

    file_name_coord = os.path.join(ss.path_base, ss.csv_name_coord)
    path_file = os.path.join(ss.path_base, ss.step1_folder_name)

    label = pd.read_csv(file_name_coord, sep = ';')
    coords = np.asarray(label[["X", "Y"]], dtype=np.float64)

    path_csv = os.path.join(ss.path_base, ss.fname_points.split(".")[0] + "_binding.csv")
    df1 = pd.read_csv(path_csv, sep = ';')

    file_name_coord = os.path.join(ss.path_base, ss.csv_name_coord)
    df2 = pd.read_csv(file_name_coord, sep = ';')

    combined_dataframe = df2.merge(df1, on= ('X', 'Y'))
    combined_dataframe.to_csv(os.path.join(ss.path_base, ss.fname_points.split(".")[0] + "_res.csv"), index = False, sep=';') 

    print("First step clustering (accumulating RAM)...")

    obj_ram = RAM(path_file = path_file, coordinates = coords, combined_dataframe = combined_dataframe)
    obj_ram.accumulating()

    print("Second step clustering (using RAM)...")
    obj_ram.exploitation(path_file_save)

if __name__ == "__main__" :
    ss = SS()
    yml_path = "settings\settings.yaml"
    ss.set(yml_path)
    segmentation_ram(ss)



# for fname in tqdm(os.listdir(path_file)):
#     if fname.endswith('.pcd'):

#         pc_tree = PCD_TREE()
#         pc_tree.open(os.path.join(path_file, fname))

#         P = pd.DataFrame(pc_tree.points, columns = ['X','Y','Z'])
#         X = np.asarray(P)
#         clustering = DBSCAN(eps=0.5, min_samples=50).fit(X)
#         labels=clustering.labels_

#         centers_labels = []
#         l_points = []
#         for i in np.unique(labels):
#             if i>-1:
#                 idx_layer=np.where(labels==i)
#                 i_data = pc_tree.points[idx_layer]
#                 center = PCD_UTILS.center_m(i_data[:,0:2])
#                 lowest_point = i_data[np.argmin(i_data[:,2])]
#                 centers_labels.append(center)
#                 l_points.append(lowest_point)

#         centers_labels = np.asarray(centers_labels)
#         main_cluster = np.argmin(np.array(l_points)[:,2])

#         if np.unique(labels).shape[0]>2:

#             x_value = combined_dataframe.loc[combined_dataframe['Name_tree'] == fname, 'X'].values[0]
#             y_value = combined_dataframe.loc[combined_dataframe['Name_tree'] == fname, 'Y'].values[0]

#             points_of_trees = coords[np.all(abs(coords - [x_value, y_value])<10, axis=1)]  
#             points_of_trees = np.delete(points_of_trees, np.all(abs(points_of_trees - [x_value, y_value])<0.0001, axis=1), axis=0)
#             centers_labels = np.delete(centers_labels, main_cluster, axis = 0)
#             distances = cdist(points_of_trees[:,0:2], centers_labels)
#             labels_indices = np.argmin(distances, axis=0)

#             XP = pd.DataFrame(pc_tree.points, columns = ['X','Y','Z'])
#             XP['I'] = pc_tree.intensity
#             XP = np.asarray(XP)

#             ci = 0
#             for c in np.unique(labels):
#                 if ((c != -1)&(c != main_cluster)):
#                     i_layer=np.where(labels==c)
#                     c_points = XP[i_layer]
#                     np_c_points = np.asarray(c_points)
#                     name = combined_dataframe.loc[(abs(combined_dataframe['X'] - points_of_trees[labels_indices][ci][0]) < 0.0001) & (abs(combined_dataframe['Y'] - points_of_trees[labels_indices][ci][1]) < 0.0001), 'Name_tree'].values[0]
#                     ids = combined_dataframe.index[combined_dataframe['Name_tree'] == name].tolist()

#                     labels_indices_list = np.arange(np_c_points.shape[0], dtype=int)
#                     labels_indices_list = np.full_like(labels_indices_list, ids[0])
                    
#                     myRAM_l = [list(point) + [label] for point, label in zip(c_points, labels_indices_list)]
#                     myRAM_l = np.asarray(myRAM_l)
#                     myRAM_list = np.concatenate((myRAM_list, myRAM_l), axis=0)
#                     ci += 1

# myRAM_list = np.delete(myRAM_list, 0, axis=0)
# myRAM = pd.DataFrame(myRAM_list, columns=['X', 'Y', 'Z', 'I', 'L'])


# for fname in tqdm(os.listdir(path_file)):
#     if fname.endswith('.pcd'):

#         pc_tree = PCD_TREE()
#         pc_tree.open(os.path.join(path_file, fname))
        
#         x_value = combined_dataframe.loc[combined_dataframe['Name_tree'] == fname, 'X'].values[0]
#         y_value = combined_dataframe.loc[combined_dataframe['Name_tree'] == fname, 'Y'].values[0]

#         name = combined_dataframe.loc[(abs(combined_dataframe['X'] - x_value) < 0.0001) & (abs(combined_dataframe['Y'] - y_value) < 0.0001), 'Name_tree'].values[0]
#         ids = combined_dataframe.index[combined_dataframe['Name_tree'] == name].tolist()
#         data_from_ram = myRAM.loc[myRAM['L'] == ids[0]]
#         data_from_ram = np.asarray(data_from_ram)

#         P = pd.DataFrame(pc_tree.points, columns = ['X','Y','Z'])
#         X = np.asarray(P)

#         clustering = DBSCAN(eps=0.5, min_samples=50).fit(X)
#         labels=clustering.labels_

#         XP = pd.DataFrame(pc_tree.points, columns = ['X','Y','Z'])
#         XP['I'] = pc_tree.intensity
#         XP = np.asarray(XP)

#         l_points = []
#         for i in np.unique(labels):
#             if i>-1:
#                 idx_layer=np.where(labels==i)
#                 i_data = pc_tree.points[idx_layer]
#                 highest_point = i_data[np.argmax(i_data[:,2])]
#                 lowest_point = i_data[np.argmin(i_data[:,2])]
#                 l_points.append(lowest_point)


#         main_cluster = np.argmin(np.array(l_points)[:,2])

#         idx_l = np.where(labels==main_cluster)
#         cur_points = pc_tree.points[idx_l]
#         cur_points_intensity = pc_tree.intensity[idx_l]

#         cur_points = np.asarray(cur_points)
#         cur_points_intensity = np.asarray(cur_points_intensity)
    
#         filename = f"{fname}"
#         if cur_points.shape[0]>100:
#             dt = np.c_[cur_points, cur_points_intensity]
#             if data_from_ram.shape[0]>0:
#                 data_from_ram = data_from_ram[:, :-1]
#                 dt = np.concatenate((dt, data_from_ram), axis=0)
#             dt = np.array(dt, dtype=np.float32)
#             pc_out = PCD(points = dt[:,0:3], intensity = dt[:,3])
#             pc_out.save(os.path.join(path_file_save, filename))

