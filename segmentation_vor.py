
from classes.PCD_AREA import PCD_AREA
from classes.PCD_TREE import PCD_TREE
import os
import numpy as np
from tqdm import tqdm
from classes.PCD_UTILS import PCD_UTILS
import pandas as pd
from settings.seg_settings import SS
from shapely.geometry import Polygon


def makedirs_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_binding_file(pc_area, ss):
    path_csv = os.path.join(ss.path_base, ss.fname_points.split(".")[0] + "_binding.csv")
    df = pd.DataFrame({"Name_tree": [], "X": [], "Y": []})
    i=0
    print("Make binding file ...")
    for polygon in tqdm(pc_area.polygons):
        i+=1
        pc_poly = pc_area.poly_cut(polygon, mode = 'main', returned = 'tree')
        if pc_poly.points.shape[0]>0:
            filename_out = str(i).rjust(4, '0') + '.pcd'
            filename_out = f"tree_{filename_out}"
            df = df.append({"Name_tree": filename_out, "X": pc_poly.coordinate[0], "Y": pc_poly.coordinate[1]}, ignore_index=True)
    df.to_csv(path_csv, index=False, sep=';')


def segmentation_vor(ss, make_binding = True):
    path_file_save = os.path.join(ss.path_base, ss.step1_folder_name)
    makedirs_if_not_exist(path_file_save)

    file_name_coord = os.path.join(ss.path_base, ss.csv_name_coord)
    file_name_data = os.path.join(ss.path_base, ss.fname_points)
    file_shape = os.path.join(ss.path_base, ss.fname_shape) 

    label = pd.read_csv(file_name_coord, sep = ';')
    coords = np.asarray(label[["X", "Y"]], dtype=np.float64)


    pc_area = PCD_AREA()
    pc_area.open(file_name_data, verbose = True)
    pc_area.unique()
    pc_area.coordinates = coords
    try:
        shp_poly = PCD_UTILS.shp_open(file_shape)
    except:
        shp_poly = PCD_UTILS.shp_create(pc_area)

    pc_area.shp_ply = Polygon(shp_poly)
    # #
    # pc_area = pc_area.poly_cut(shp_poly, returned = 'area')
    # pc_area.save(os.path.join(ss.path_base, "loc1_cut.pcd"))
    # #
    pc_area.vor_regions(verbose = False)
    
    if make_binding:
        make_binding_file(pc_area, ss)

    i=0
    print("Start polygons processing ...")
    for polygon in tqdm(pc_area.polygons):
        i+=1
        if i<ss.first_num:
            continue
        pc_poly = pc_area.poly_cut(polygon, mode = 'main')
        if pc_poly.points.shape[0]>0:

            LOW = pc_poly.points.min(axis=0)[2]
            STEP = ss.STEP
            HIGH = LOW + STEP
            z_thresholds = np.array(ss.z_thresholds)
            eps_steps = np.array(ss.eps_steps)
            min_pts = np.array(ss.min_pts)

            offsetX, offsetY = 0,0
            old_uc = pc_poly.coordinate
            old_lc = pc_poly.coordinate
            j = 0

            filename_out = str(i).rjust(4, '0') + '.pcd'
            filename_out = f"tree_{filename_out}"

            for zc in tqdm(range(2*int(pc_poly.points.max(axis=0)[2]//STEP))):
                idx = np.searchsorted(z_thresholds * pc_poly.points.max(axis=0)[2], min(LOW,pc_poly.points.max(axis=0)[2]), side='left')
                eps_step = eps_steps[idx]
                min_pt = min_pts[idx]
    
                pc_l_p = pc_area.make_layer_polygon(polygon, offsetX, offsetY, pc_poly.coordinate, LOW, HIGH)
                pc_l_p.lower_coordinate = [old_uc[0], old_uc[1], (old_lc[2]+old_uc[2])/2]
                pc_l_p.process_layer(0.35+eps_step, min_pt, verbose = False)
                old_lc = pc_l_p.lower_coordinate
                old_uc = pc_l_p.upper_coordinate
                offsetX, offsetY = pc_l_p.offset[0], pc_l_p.offset[1]

                LOW = LOW + STEP/2
                HIGH = LOW + STEP

                if pc_l_p.points.shape[0]>1:
                    if j==0:
                        result_points = np.copy(pc_l_p.points)
                        result_intensity = np.copy(pc_l_p.intensity)
                    else: 
                        result_points = np.vstack((result_points, pc_l_p.points))
                        result_intensity = np.hstack((result_intensity, pc_l_p.intensity))
                    j += 1

            pc_result = PCD_TREE(points = result_points, intensity = result_intensity, coordinate = pc_poly.coordinate)
            pc_result.unique()
            file_name_data_out = os.path.join(path_file_save, filename_out) 
            pc_result.save(file_name_data_out)

if __name__ == "__main__" :
    ss = SS()
    yml_path = "settings\settings.yaml"
    ss.set(yml_path)
    segmentation_vor(ss, make_binding = False)
