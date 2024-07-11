from classes.PCD import PCD
from classes.PCD_AREA import PCD_AREA
from classes.PCD_TREE import PCD_TREE
import os
import numpy as np
from tqdm import tqdm
from classes.PCD_UTILS import PCD_UTILS
import pandas as pd


source_folder = 'D:\\Paulava_Monumentse57\\Carbon_Polygon_Loc_1\\mat\\v5_CPL1\\copy0'
species_labels_path = "D:\\Paulava_Monumentse57\\Carbon_Polygon_Loc_1\\mat\\species.csv"
path_file_save = 'D:\\Paulava_Monumentse57\\Carbon_Polygon_Loc_1\\mat\\v5_CPL1\\copy0\\after_species1'
species_labels = pd.read_csv(species_labels_path, sep = ';')

for file_name in os.listdir(source_folder):
    if file_name.endswith('.pcd'):
        source_file_path = os.path.join(source_folder, file_name)
        pc_tree = PCD_TREE()
        pc_tree.open(source_file_path, mode = 'rgb')

        species = species_labels.loc[species_labels['Name_tree'] == file_name, 'label'].values[0]
        
        LOW = pc_tree.points.min(axis=0)[2]
        STEP = 2.5
        HIGH = LOW + STEP
        j = 0

        for zc in range(2*int(pc_tree.points.max(axis=0)[2]//STEP)):

            pc_layer = PCD_AREA(pc_tree.points, pc_tree.intensity)
            idx_labels=np.where((pc_layer.points[:,2]>LOW)&(pc_layer.points[:,2]<=HIGH))
            pc_layer.index_cut(idx_labels)

            if species == '0':
                idx_labels=np.where(pc_layer.intensity>=0)
                if LOW < 20:
                    idx_labels=np.where(pc_layer.intensity>=2000)
                if LOW < 15:
                    idx_labels=np.where(pc_layer.intensity>=5000)
                if LOW < 10:
                    idx_labels=np.where(pc_layer.intensity>=7000)
                if LOW < 5:
                    idx_labels=np.where(pc_layer.intensity>=10000)
                pc_layer.index_cut(idx_labels)
            
            LOW = LOW + STEP
            HIGH = HIGH + STEP
        
            if pc_layer.points.shape[0]>1:
                if j==0:
                    result_points = np.copy(pc_layer.points)
                    result_intensity = np.copy(pc_layer.intensity)
                else: 
                    result_points = np.vstack((result_points, pc_layer.points))
                    result_intensity = np.hstack((result_intensity, pc_layer.intensity))
                j += 1
        
        pc_result = PCD_TREE(points = result_points, intensity = result_intensity)
        pc_result.unique()

        file_name_data_out = os.path.join(path_file_save, file_name) 
        pc_result.save(file_name_data_out)

