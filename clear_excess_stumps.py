import os
import pandas as pd
import numpy as np
from settings.coord_settings import CS
import shutil
import predict
from tqdm import tqdm

def count_num_files(cs):
    txt_path = os.path.join(cs.path_base, "coordinates_paths.txt") 
    file = open(txt_path, "r")
    i = 0
    while True:
        line = file.readline()
        if not line:
            break
        if line.strip() == '':
            continue
        i += 1
    file.close()
    return i

def makedirs_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def clear_excess_stumps(cs):
    model_name = 'int0000_7000-512-rlish-s4762'
    pth = os.path.join(cs.path_base, cs.fname_points.partition('.')[0] + "_Coordinates_Merged.csv")
    df = pd.read_csv(pth, delimiter=";")

    names_col = []
    n = count_num_files(cs)
    path_merged = os.path.join(cs.path_base, "merged")
    makedirs_if_not_exist(path_merged)
    first_n_columns = df.iloc[:, :n]
    column_names = first_n_columns.columns
    initial_labels = np.full((1, df.shape[0]),-1)
    for i in tqdm(range(n)):
        labels = []
        parts = column_names[i].split("_")
        parts_int = parts[-1]
        if "." in parts_int:
            parts_int = parts_int.split(".")[0]
        names_col.append("Labels_"+str(parts_int))
        path_int = os.path.join(cs.path_base, parts_int, cs.cut_data_method + '_cells', 'stumps')
        for j in tqdm(range(df.shape[0])):
            value = first_n_columns.at[j, column_names[i]]
            if value != "File__Not__Found":
                path_file = os.path.join(path_int, value)
                path_save = os.path.join(path_merged, value)
                try:
                    shutil.copy2(path_file, path_save)

                    label = predict.test(path_save, model_name)
                    labels.append(label)   
                except FileNotFoundError:
                    print(f"No such file: {path_file}")
                    labels.append(-3)
            elif value == "File__Not__Found":
                labels.append(-2)
            else:
                print("ERROR")
                break
        labels = np.asarray([labels])
        initial_labels = np.vstack([initial_labels, labels])
    initial_labels = initial_labels.T

    df_labels = pd.DataFrame(data = initial_labels[:,1:n+1], columns=names_col)
    df_result = pd.concat([df, df_labels], axis=1)

    save_pth = cs.fname_points.partition('.')[0] + "_Clear_Excess.csv"  
    save_pth = os.path.join(cs.path_base, save_pth)
    df_result.to_csv(save_pth, index = False, sep=';')

if __name__ == "__main__" :
    cs = CS()
    yml_path = "settings\settings.yaml"
    cs.set(yml_path)
    clear_excess_stumps(cs)
