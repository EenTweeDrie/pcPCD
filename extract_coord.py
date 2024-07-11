import pandas as pd
import os
from settings.coord_settings import CS


def extract_coord(cs):
    pth = cs.fname_points.partition('.')[0] + "_Clear_Excess.csv"
    pth = os.path.join(cs.path_base, pth)
    data = pd.read_csv(pth, sep=';')
    num_label_columns = sum(['Labels' in col for col in data.columns])
    for dm in range(num_label_columns):  # detection min
        num_ones = data[[col for col in data.columns if 'Labels' in col]].eq(1).sum(axis=1)
        filtered_data = data[num_ones >= dm + 1]

        save_pth = cs.fname_points.partition('.')[0] + "_Extract_Coordinates_dm=" + str(dm + 1) + ".csv"
        save_pth = os.path.join(cs.path_base, save_pth)
        filtered_data.to_csv(save_pth, index=False, sep=';')

    for dc in range(num_label_columns + 1):  # detection count
        num_ones = data[[col for col in data.columns if 'Labels' in col]].eq(1).sum(axis=1)
        filtered_data = data[num_ones == dc]

        save_pth = cs.fname_points.partition('.')[0] + "_Extract_Coordinates_dc=" + str(dc) + ".csv"
        save_pth = os.path.join(cs.path_base, save_pth)
        filtered_data.to_csv(save_pth, index=False, sep=';')


if __name__ == "__main__":
    cs = CS()
    yml_path = "settings\settings.yaml"
    cs.set(yml_path)
    extract_coord(cs)
