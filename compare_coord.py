import os
import numpy as np
import pandas as pd
from classes.PCD import PCD
from settings.coord_settings import CS
import glob
import math

def compare_coord(cs, eps = 0.3):
    file_path = "E:/Paulava_Monumentse57/Carbon_Polygon_Loc_1/vkrm_pic/full/Loc_2/Loc2_Local.pcd"

    file_pattern = cs.path_base + '/*_Extract_Coordinates_dm=*.csv'
    file_list = glob.glob(file_pattern)
    num_files = len(file_list)
    
    for nf in range(num_files):

        pc_t = PCD()
        pc_t.open(file_path)
        points = pc_t.points[:,0:2]

        csv_path = cs.fname_points.partition('.')[0] + "_Extract_Coordinates_dm=" + str(nf+1) + ".csv"
        csv_path = os.path.join(cs.path_base, csv_path)
        my_data = pd.read_csv(csv_path, sep =';')
        my_points = np.asarray(my_data[["X", "Y"]], dtype=np.float64)

        points_0 = [] 
        my_points_0 = [] 
        NE = 0
        j = -1
        for m in points:
            tc = False
            i = -1
            j += 1
            for p in my_points: 
                i += 1
                if math.sqrt((p[0] - m[0])**2 + (p[1] - m[1])**2)<eps:
                    NE+=1
                    points = np.delete(points,(j), axis = 0)
                    my_points = np.delete(my_points,(i), axis = 0)
                    # my_points_names = np.delete(my_points_names,(i), axis = 0)
                    my_points_0.append(m)
                    points_0.append(p)
                    i -=1
                    j -= 1
                    tc = True
                if tc:
                    break

        points_0 = np.asarray(points_0)
        my_points_0 = np.asarray(my_points_0)
        points = np.asarray(points)
        my_points = np.asarray(my_points)

        print(f"dm={nf+1}", points_0.shape[0], points.shape[0], my_points_0.shape[0], my_points.shape[0], f"eps={eps:.2f}")

        filename = os.path.join(cs.path_base, 'LP', f'poteryannie_dm={nf+1}_eps={eps:.2f}.csv')
        np.savetxt(filename, points, delimiter=";", fmt="%.6f")

        filename = os.path.join(cs.path_base, 'LP', f'lishnie_dm={nf+1}_eps={eps:.2f}.csv')
        a_tolist = my_points.tolist()
        a_tolist = np.asarray(a_tolist)
        np.savetxt(filename, a_tolist, delimiter=";", fmt="%.6f")

    pc_t = PCD()
    pc_t.open(file_path)
    points = pc_t.points[:,0:2]

    csv_path = cs.fname_points.partition('.')[0] + "_Clear_Excess.csv"
    csv_path = os.path.join(cs.path_base, csv_path)
    my_data = pd.read_csv(csv_path, sep =';')
    my_points = np.asarray(my_data[["X", "Y"]], dtype=np.float64)

    points_0 = [] 
    my_points_0 = [] 
    NE = 0
    j = -1
    for m in points:
        tc = False
        i = -1
        j += 1
        for p in my_points: 
            i += 1
            if math.sqrt((p[0] - m[0])**2 + (p[1] - m[1])**2)<eps:
                NE+=1
                points = np.delete(points,(j), axis = 0)
                my_points = np.delete(my_points,(i), axis = 0)
                # my_points_names = np.delete(my_points_names,(i), axis = 0)
                my_points_0.append(m)
                points_0.append(p)
                i -=1
                j -= 1
                tc = True
            if tc:
                break

    points_0 = np.asarray(points_0)
    my_points_0 = np.asarray(my_points_0)
    points = np.asarray(points)
    my_points = np.asarray(my_points)

    print(f"dm={0}",points_0.shape[0], points.shape[0], my_points_0.shape[0], my_points.shape[0], f"eps={eps:.2f}")

    filename = os.path.join(cs.path_base, 'LP', f'poteryannie_dm={0}_eps={eps}.csv')
    np.savetxt(filename, points, delimiter=";", fmt="%.6f")

    filename = os.path.join(cs.path_base, 'LP', f'lishnie_dm={0}_eps={eps}.csv')
    a_tolist = my_points.tolist()
    a_tolist = np.asarray(a_tolist)
    np.savetxt(filename, a_tolist, delimiter=";", fmt="%.6f")



    # file_pattern = cs.path_base + '/*_Extract_Coordinates_dc=*.csv'
    # file_list = glob.glob(file_pattern)
    # num_files = len(file_list)
    
    # for nf in range(num_files):

    #     pc_t = PCD()
    #     pc_t.open(file_path)
    #     points = pc_t.points[:,0:2]

    #     csv_path = cs.fname_points.partition('.')[0] + "_Extract_Coordinates_dc=" + str(nf) + ".csv"
    #     csv_path = os.path.join(cs.path_base, csv_path)
    #     my_data = pd.read_csv(csv_path, sep =';')
    #     my_points = np.asarray(my_data[["X", "Y"]], dtype=np.float64)

    #     points_0 = [] 
    #     my_points_0 = [] 
    #     NE = 0
    #     j = -1
    #     for m in points:
    #         tc = False
    #         i = -1
    #         j += 1
    #         for p in my_points: 
    #             i += 1
    #             if math.sqrt((p[0] - m[0])**2 + (p[1] - m[1])**2)<eps:
    #                 NE+=1
    #                 points = np.delete(points,(j), axis = 0)
    #                 my_points = np.delete(my_points,(i), axis = 0)
    #                 # my_points_names = np.delete(my_points_names,(i), axis = 0)
    #                 my_points_0.append(m)
    #                 points_0.append(p)
    #                 i -=1
    #                 j -= 1
    #                 tc = True
    #             if tc:
    #                 break

    #     points_0 = np.asarray(points_0)
    #     my_points_0 = np.asarray(my_points_0)
    #     points = np.asarray(points)
    #     my_points = np.asarray(my_points)

    #     print(f"dc={nf}", points_0.shape[0], points.shape[0], my_points_0.shape[0], my_points.shape[0])

    #     filename = os.path.join(cs.path_base, f'poteryannie_dc={nf}.csv')
    #     np.savetxt(filename, points, delimiter=";", fmt="%.6f")

    #     filename = os.path.join(cs.path_base, f'lishnie_dc={nf}.csv')
    #     a_tolist = my_points.tolist()
    #     a_tolist = np.asarray(a_tolist)
    #     np.savetxt(filename, a_tolist, delimiter=";", fmt="%.6f")





if __name__ == "__main__" :
    cs = CS()
    yml_path = "settings\settings.yaml"
    cs.set(yml_path)
    eps=0
    while eps<0.6:
        if eps == 0:
            compare_coord(cs, 0.025)
        else: 
            compare_coord(cs, eps)
        eps = eps + 0.05