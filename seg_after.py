from classes.PCD import PCD
from classes.PCD_AREA import PCD_AREA
from classes.PCD_TREE import PCD_TREE
import os
import numpy as np
from tqdm import tqdm
from classes.PCD_UTILS import PCD_UTILS
import pandas as pd
import settings.seg_settings as ss

def makedirs_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def seg_after(model_name):

    source_folder = os.path.join(ss.path_base, ss.step1_folder_name, ss.step2_folder_name, ss.step3_folder_name)
    species_labels_path = os.path.join(source_folder,'predict_' + model_name + '.csv')
    path_file_save = os.path.join(source_folder, model_name)
    makedirs_if_not_exist(path_file_save)

    species_labels = pd.read_csv(species_labels_path, sep = ';')
    print(species_labels)

    for file_name in tqdm(os.listdir(source_folder)):
        if file_name.endswith('.pcd'):
            source_file_path = os.path.join(source_folder, file_name)
            pc_tree = PCD_TREE()
            pc_tree.open(source_file_path)

            species = species_labels.loc[species_labels['Name_tree'] == file_name, 'Label'].values[0]
            
            LOW = pc_tree.points.min(axis=0)[2]
            STEP = 2.5
            HIGH = LOW + STEP
            j = 0

            for zc in range(2*int(pc_tree.points.max(axis=0)[2]//STEP)):

                pc_layer = PCD_AREA(pc_tree.points, pc_tree.intensity)
                idx_labels=np.where((pc_layer.points[:,2]>LOW)&(pc_layer.points[:,2]<=HIGH))
                pc_layer.index_cut(idx_labels)

                if species == 0:
                    idx_labels=np.where(pc_layer.intensity>=0)
                    if LOW < 20:
                        idx_labels=np.where(pc_layer.intensity>=1000)
                    if LOW < 15:
                        idx_labels=np.where(pc_layer.intensity>=4000)
                    if LOW < 10:
                        idx_labels=np.where(pc_layer.intensity>=6000)
                    if LOW < 5:
                        idx_labels=np.where(pc_layer.intensity>=9000)
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

if __name__ == '__main__':
    model_name = 'v5_cpl1-1024-rvc-s1024'
    seg_after(model_name)  
