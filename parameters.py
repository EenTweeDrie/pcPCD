from settings.seg_settings import SS
import os
from tqdm import tqdm
from classes.PCD_TREE import PCD_TREE
import pandas as pd

def parameters(ss, path_file, K = 0):
    param_name = ss.fname_points.partition('.')[0] + "_Parameters.csv"  
    path_save = os.path.join(ss.path_base, param_name)
    mi = 0
    path_csv = os.path.join(ss.path_base, ss.fname_points.split(".")[0] + "_res.csv")
    df = pd.read_csv(path_csv, sep = ';')
    
    names = []
    diameters_ls = []
    diameters_hls = []
    heights = []
    lenghts = []
    crown_v = []
    crown_area = []
    xy_crown_area = []
    xz_crown_area = []
    yz_crown_area = []
    x_up = []
    y_up = []
    x_up1 = []
    y_up1 = []
    x_up2 = []
    y_up2 = []
    x_up3 = []
    y_up3 = []

    for fname in tqdm(os.listdir(path_file)):
        if fname.endswith('.pcd'):
            mi += 1
            if mi < K:
                continue
        
            pc_tree = PCD_TREE()
            pc_tree.open(os.path.join(path_file, fname))
            pc_tree.RGBint = pc_tree.intensity/max(pc_tree.intensity)
            
            pc_tree.estimate_height()
            pc_tree.estimate_length()
            pc_tree.search_main_coordinate(df, fname)
            pc_slice = pc_tree.search_slice()
            pc_expsph = pc_tree.search_points_for_center(pc_slice)
            if pc_expsph.points.shape[0]>10:
                pc_tree.estimate_diameter(pc_expsph, pc_slice)

                points_no_trunk = pc_tree.search_points_no_trunk()
                if points_no_trunk.shape[0]>0:
                    pc_tree.estimate_crown(points_no_trunk)
                else:
                    pc_tree.crown_volume, pc_tree.crown_volume, pc_tree.crown_square, pc_tree.xy_crown_square, pc_tree.yz_crown_square, pc_tree.xz_crown_square = 0, 0, 0, 0, 0, 0
            else:
                pc_tree.height = 0
                pc_tree.length = 0
                pc_tree.diameter_LS = 0
                pc_tree.diameter_HLS = 0
                pc_tree.crown_volume, pc_tree.crown_volume, pc_tree.crown_square, pc_tree.xy_crown_square, pc_tree.yz_crown_square, pc_tree.xz_crown_square = 0, 0, 0, 0, 0, 0
            
            down_point = pc_tree.points.max(axis=0)[2]-2
            pc_upslice_2m = pc_tree.search_up_slice(down_point)
            pc_tree.search_up_coord(pc_upslice_2m)

            names.append(fname)
            diameters_ls.append(pc_tree.diameter_LS)
            diameters_hls.append(pc_tree.diameter_HLS)
            heights.append(pc_tree.height)
            lenghts.append(pc_tree.length)
            crown_v.append(pc_tree.crown_volume)
            crown_area.append(pc_tree.crown_square)
            xy_crown_area.append(pc_tree.xy_crown_square)
            yz_crown_area.append(pc_tree.yz_crown_square)
            xz_crown_area.append(pc_tree.xz_crown_square)
            x_up.append(pc_tree.x_up)
            y_up.append(pc_tree.y_up)

            # print(fname, pc_tree.diameter_LS, pc_tree.diameter_HLS, pc_tree.height, pc_tree.length, pc_tree.crown_volume, pc_tree.crown_square,
            # pc_tree.xy_crown_square, pc_tree.yz_crown_square, pc_tree.xz_crown_square, pc_tree.x_up, pc_tree.y_up)

            down_point = 2*(pc_tree.points.max(axis=0)[2]-pc_tree.points.min(axis=0)[2])/3+pc_tree.points.min(axis=0)[2]
            pc_upslice_third = pc_tree.search_up_slice(down_point)
            pc_tree.search_up_coord(pc_upslice_third)
  
            x_up1.append(pc_tree.x_up)
            y_up1.append(pc_tree.y_up)

            pc_tree.search_up_coord(pc_upslice_third, mode = 'median')

            x_up2.append(pc_tree.x_up)
            y_up2.append(pc_tree.y_up)

            pc_tree.search_up_coord(pc_upslice_2m, mode = 'highest')

            x_up3.append(pc_tree.x_up)
            y_up3.append(pc_tree.y_up)


    bd = pd.DataFrame({"Name": names,"Diameter_LS, cm": diameters_ls,"Diameter_HLS, cm": diameters_hls,"Height, m": heights,"Length, m": lenghts, 
    "Crown_volume, m3": crown_v, "Crown_square, m2": crown_area, "XY_crown_square, m2": xy_crown_area, "XZ_crown_square, m2": xz_crown_area, "YZ_crown_square, m2": yz_crown_area,
    "X_UP_0": x_up, "Y_UP_0": y_up,
    "X_UP_1": x_up1, "Y_UP_1": y_up1,
    "X_UP_2": x_up2, "Y_UP_2": y_up2,
    "X_UP_3": x_up3, "Y_UP_3": y_up3})
    bd.to_csv(path_save, index = False, sep=';')     

if __name__ == "__main__" :
    ss = SS()
    yml_path = "settings\settings.yaml"
    ss.set(yml_path)
    path_file = os.path.join(ss.path_base, ss.step1_folder_name, ss.step2_folder_name, ss.step3_folder_name)
    parameters(ss, path_file)



                
