import os
import pandas as pd
import numpy as np
from settings.coord_settings import CS

def makedirs_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def merge(file1, file2, iter, names_col, array):
    array = np.asarray(array)

    n_col_name = 0 + iter
    n_col_diam = (iter+1)*2

    eps = 0.25
    for index, row in file1.iterrows():
        array = np.vstack([array, np.asarray(row)])

    array = array[1:]
    XY = array[:,iter:iter+2]

    added_column_name = np.asarray(np.full(array.shape[0],"File__Not__Found"), dtype=str)
    added_column_diameter = np.asarray(np.full(array.shape[0],0.0), dtype=np.float32)
    for index, row in file2.iterrows():
        for point in XY:
            if np.linalg.norm(np.asarray([float(point[0]), float(point[1])]) - np.asarray([float(row[1]), float(row[2])])) < eps:
                idx = np.where(XY == point)[0][0]
                
                added_column_name[idx] = row[0]
                added_column_diameter[idx] = row[3]

    array = np.insert(array, n_col_diam, added_column_diameter, axis=1)
    array = np.insert(array, n_col_name, added_column_name, axis=1)
   

    for index, row in file2.iterrows():
        AddFlag = True
        for point in XY:
            if np.linalg.norm(np.asarray([float(point[0]), float(point[1])]) - np.asarray([float(row[1]), float(row[2])])) < eps:
                AddFlag = False
        if AddFlag:
            added_row = ["File__Not__Found",row[0],row[1],row[2],0.0,row[3]]
            if iter > 1:
                added_row.insert((iter)*2, 0.0)
                added_row.insert(iter-1, "File__Not__Found")
            added_row = np.asarray(added_row)
            array = np.vstack([array, added_row])

    df = pd.DataFrame(data = array, columns=names_col)
    df = df.dropna()
    df = df[(df.X != 'nan')]
    df = df[(df.Y != 'nan')]
    return df

def init_merge_file(cs):
    txt_path = os.path.join(cs.path_base, "coordinates_paths.txt") 
    file = open(txt_path, "r")
    i = 0
    iter = 0
    while True:
        line = file.readline()
        if not line:
            break
        file_name = line.strip()
        if file_name == '':
            continue
        file1_path = file_name
        splt_fn = file_name.split(sep="_")[-1]
        splt_fn = splt_fn.split(sep=".")[0]

        if i == 0:
            names_col = ["Name_stump_" + splt_fn, "X", "Y", "Diameter_" + splt_fn]
        if i > 0:
            names_col.insert((iter+2)*2, "Diameter_" + splt_fn)
            names_col.insert(iter+1, "Name_stump_" + splt_fn)
            if i == 1:
                array = ['n',0,0,0]
            else:
                array.insert(i,'n')
                array.insert(i*2+1,0)
       
        i+=1
        if i >= 2 :
            iter += 1
            if iter == 1:
                file1 = pd.read_csv(file2_path, delimiter=";")
                file2 = pd.read_csv(file1_path, delimiter=";")
            else:
                file1 = df
                file2 = pd.read_csv(file1_path, delimiter=";")
            df = merge(file1, file2, iter, names_col, [array])
        
        file2_path = file1_path

    file.close()

    return df

def merge_coordinates(cs):
    df = init_merge_file(cs)
    save_pth = cs.fname_points.partition('.')[0] + "_Coordinates_Merged.csv"  
    save_pth = os.path.join(cs.path_base, save_pth)
    df.to_csv(save_pth, index = False, sep=';')
    
if __name__ == "__main__" :
    cs = CS()
    yml_path = "settings\settings.yaml"
    cs.set(yml_path)
    merge_coordinates(cs)
