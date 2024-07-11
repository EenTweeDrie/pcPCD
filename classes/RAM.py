import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.cluster import DBSCAN
import os
from scipy.spatial.distance import cdist
from tqdm import tqdm
from classes.PCD_TREE import PCD_TREE
from classes.PCD_UTILS import PCD_UTILS
from classes.PCD import PCD

class RAM():
    def __init__(self, path_file, coordinates, combined_dataframe, ram = None):
        self.path_file = path_file
        self.coordinates = coordinates
        self.combined_dataframe = combined_dataframe
        self.ram = ram

    def search_labels(pc_tree, cluster_labels):
        centers_labels = []
        l_points = []
        for i in np.unique(cluster_labels):
            if i>-1:
                idx_layer=np.where(cluster_labels==i)
                i_data = pc_tree.points[idx_layer]
                center = PCD_UTILS.center_m(i_data[:,0:2])
                lowest_point = i_data[np.argmin(i_data[:,2])]
                centers_labels.append(center)
                l_points.append(lowest_point)
        return centers_labels, l_points
    
    def clustering(pc_tree):
        P = pd.DataFrame(pc_tree.points, columns = ['X','Y','Z'])
        X = np.asarray(P)
        # clustering = DBSCAN(eps=0.5, min_samples=50).fit(X)
        clustering = DBSCAN(eps=0.65, min_samples=50).fit(X)
        labels=clustering.labels_
        return labels
    
    def get_xy_from_df(self, fname):
        x_value = self.combined_dataframe.loc[self.combined_dataframe['Name_tree'] == fname, 'X'].values[0]
        y_value = self.combined_dataframe.loc[self.combined_dataframe['Name_tree'] == fname, 'Y'].values[0]
        return x_value, y_value
    
    def get_idnames_from_df(self, x_value, y_value):
        name = self.combined_dataframe.loc[(abs(self.combined_dataframe['X'] - x_value) < 0.0001) & (abs(self.combined_dataframe['Y'] - y_value) < 0.0001), 'Name_tree'].values[0]
        idnames = self.combined_dataframe.index[self.combined_dataframe['Name_tree'] == name].tolist()
        return idnames

    def accumulating(self):
        myRAM_list = [[0,0,0,0,0]]
        for fname in tqdm(os.listdir(self.path_file)):
            if fname.endswith('.pcd'):

                pc_tree = PCD_TREE()
                pc_tree.open(os.path.join(self.path_file, fname))

                labels = RAM.clustering(pc_tree)
                
                centers_labels, l_points = RAM.search_labels(pc_tree, labels)

                centers_labels = np.asarray(centers_labels)
                try:
                    main_cluster = np.argmin(np.array(l_points)[:,2])
                except:
                    main_cluster = -1

                if np.unique(labels).shape[0]>2:

                    x_value, y_value = self.get_xy_from_df(fname)

                    points_of_trees = self.coordinates[np.all(abs(self.coordinates - [x_value, y_value])<10, axis=1)]
                    points_of_trees = np.delete(points_of_trees, np.all(abs(points_of_trees - [x_value, y_value])<0.0001, axis=1), axis=0)
                    centers_labels = np.delete(centers_labels, main_cluster, axis = 0)
                    distances = cdist(points_of_trees[:,0:2], centers_labels)
                    labels_indices = np.argmin(distances, axis=0)

                    XP = pd.DataFrame(pc_tree.points, columns = ['X','Y','Z'])
                    XP['I'] = pc_tree.intensity
                    XP = np.asarray(XP)

                    ci = 0
                    for c in np.unique(labels):
                        if ((c != -1)&(c != main_cluster)):
                            i_layer=np.where(labels==c)
                            c_points = XP[i_layer]
                            np_c_points = np.asarray(c_points)

                            ids = self.get_idnames_from_df(points_of_trees[labels_indices][ci][0], points_of_trees[labels_indices][ci][1])

                            labels_indices_list = np.arange(np_c_points.shape[0], dtype=int)
                            labels_indices_list = np.full_like(labels_indices_list, ids[0])
                            
                            myRAM_l = [list(point) + [label] for point, label in zip(c_points, labels_indices_list)]
                            myRAM_l = np.asarray(myRAM_l)
                            myRAM_list = np.concatenate((myRAM_list, myRAM_l), axis=0)
                            ci += 1

        myRAM_list = np.delete(myRAM_list, 0, axis=0)
        self.ram = pd.DataFrame(myRAM_list, columns=['X', 'Y', 'Z', 'I', 'L'])

    def exploitation(self, path_file_save):
        for fname in tqdm(os.listdir(self.path_file)):
            if fname.endswith('.pcd'):

                pc_tree = PCD_TREE()
                pc_tree.open(os.path.join(self.path_file, fname))
                
                x_value, y_value = self.get_xy_from_df(fname)

                ids = self.get_idnames_from_df(x_value, y_value)

                data_from_ram = self.ram.loc[self.ram['L'] == ids[0]]
                data_from_ram = np.asarray(data_from_ram)

                labels = RAM.clustering(pc_tree)

                XP = pd.DataFrame(pc_tree.points, columns = ['X','Y','Z'])
                XP['I'] = pc_tree.intensity
                XP = np.asarray(XP)

                _, l_points = RAM.search_labels(pc_tree, labels)

                try:
                    main_cluster = np.argmin(np.array(l_points)[:,2])
                except:
                    main_cluster = -1

                pc_result = PCD(points = pc_tree.points, intensity = pc_tree.intensity)
                idx_l = np.where(labels==main_cluster)
                pc_result.index_cut(idx_l)
            
                filename = f"{fname}"
                if pc_result.points.shape[0]>100:
                    if data_from_ram.shape[0]>0:
                        data_from_ram = data_from_ram[:, :-1]
                        pc_result.concatenate(data_from_ram)
                    pc_result.save(os.path.join(path_file_save, filename))





