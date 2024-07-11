import os
from classes.PCD import PCD
from classes.PCD_AREA import PCD_AREA
from classes.PCD_UTILS import PCD_UTILS
from classes.CELL import CELL
from classes.VOR_TES import VOR_TES
from settings.coord_settings import CS
import numpy as np
import pandas as pd
import circle_fit as cf
import statistics
import math
from tqdm import tqdm



def makedirs_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def coordinates(intensity_cut_make, cs):
    fname_data_cut = cs.fname_points.partition('.')[0] + "_cut_int" + str(cs.intensity_cut) + ".pcd"                      # Имя создаваемого файла с обрезанными данными облака по высоте и границам участка (.pcd)
    csv_name_coord = cs.fname_points.partition('.')[0] + "_Coordinates_int" + str(intensity_cut_make) + ".csv"         # Имя создаваемого файла в папке path_base/cells/stumps/ (.csv)

    file_name_traj = os.path.join(cs.path_base, cs.fname_traj)
    file_name_data = os.path.join(cs.path_base, cs.fname_points) 
    file_shape = os.path.join(cs.path_base, cs.fname_shape) 
    file_name_data_cut = os.path.join(cs.path_base, fname_data_cut) 
    file_name_csv = os.path.join(cs.path_base, csv_name_coord) 

    if (cs.FLAG_cut_data or cs.FLAG_make_cells) and (cs.cut_data_method == 'flood_fill'):
        pc_traj = PCD()
        pc_traj.open(file_name_traj)
        pc_traj.points = PCD_UTILS.shift(pc_traj.points, cs.x_shift, cs.y_shift, cs.z_shift)

    FlagShape = True

    if cs.FLAG_cut_data:
        pc_area = PCD_AREA()
        pc_area.open(file_name_data, verbose = True)
        pc_area.points = PCD_UTILS.shift(pc_area.points, cs.x_shift, cs.y_shift, cs.z_shift)
        
        try:
            shp_poly = PCD_UTILS.shp_open(file_shape)
            shp_poly = PCD_UTILS.shift(shp_poly, cs.x_shift, cs.y_shift, cs.z_shift)
        except:
            print("Warning: File of area boundary not found. The boundaries of the area are selected as the entire loaded area.")
            FlagShape = False
            shp_poly = PCD_UTILS.shp_create(pc_area)
            

        print('Starting cutting main pcd ...')

        idx_labels=np.where((pc_area.points[:,2]>cs.LOW)&(pc_area.points[:,2]<=cs.UP))
        pc_area.index_cut(idx_labels)

        idx_labels = np.where(pc_area.intensity>=cs.intensity_cut)
        pc_area.index_cut(idx_labels)

        if FlagShape:
            pc_area = pc_area.poly_cut(shp_poly)

        pc_area.save(file_name_data_cut)

    path_int = os.path.join(cs.path_base, 'int' + str(intensity_cut_make))
    makedirs_if_not_exist(path_int)

    path_file_cells = os.path.join(cs.path_base, path_int, cs.cut_data_method + '_cells')
    makedirs_if_not_exist(path_file_cells)

    if cs.FLAG_make_cells:

        if not cs.FLAG_cut_data:
            pc_area = PCD_AREA()
            pc_area.open(file_name_data_cut)
            idx_labels = np.where(pc_area.intensity>=intensity_cut_make)
            pc_area.index_cut(idx_labels)
            pc_area.points = PCD_UTILS.shift(pc_area.points, cs.x_shift, cs.y_shift, cs.z_shift)
            try:
                shp_poly = PCD_UTILS.shp_open(file_shape)
                shp_poly = PCD_UTILS.shift(shp_poly, cs.x_shift, cs.y_shift, cs.z_shift)
            except:
                shp_poly = PCD_UTILS.shp_create(pc_area)
        
        if cs.FLAG_cut_data:
            idx_labels = np.where(pc_area.intensity>=intensity_cut_make)
            pc_area.index_cut(idx_labels)


        print('Starting extracting areas (cells) traj-based ...')

        if cs.cut_data_method == 'voronoi_tessellation':
            vortes = VOR_TES(points = pc_area.points, intensity = pc_area.intensity, algo = cs.algo, n_clusters = cs.n_clusters, intensity_cut = cs.intensity_cut_vor_tes)
            vortes.select_borders(path_file_cells, shp_poly, verbose = False)
            vortes.select_clusters(path_file_cells)
        
        elif cs.cut_data_method == 'flood_fill':
            cell = CELL(points = pc_area.points, intensity = pc_area.intensity, points_traj = pc_traj.points, cell_size = cs.cell_size)
            cell.make_cell_list(pc_area.points.min(axis=0), pc_area.points.max(axis=0), verbose = True)
            cell.save_all_cells(path_file_cells, verbose = True)

        elif cs.cut_data_method == 'none':
            path_file_stumps = os.path.join(cs.path_base, 'stumps')
            makedirs_if_not_exist(path_file_stumps)

        else:
            raise Exception("There is no such algorithm. Choose from existing: 'voronoi_tessellation', 'flood_fill', 'none'")
        
        print(f'\n {cs.n_clusters} areas (cells) have been saved to the folder {path_file_cells}')

    if cs.FLAG_make_stumps:
        
        TD = []
        TN = []
        TCX =[]
        TCY =[]

        path_file_stumps = os.path.join(path_file_cells, 'stumps')
        makedirs_if_not_exist(path_file_stumps)

        print(f'Starting stump extracting from areas (cells) ...')

        tfni = 0
        for filename in tqdm(os.listdir(path_file_cells)):
            if filename.endswith('.pcd'):
                print(f'\n Extracting from {filename} cell ...')
                if cs.cut_data_method == 'none':
                    path_cells = file_name_data
                else:
                    path_cells = os.path.join(path_file_cells, filename)

                pc_cells = CELL()
                pc_cells.open(path_cells)

                labels_stumps = pc_cells.extract_stumps_labels()

                for i in tqdm(np.unique(labels_stumps)):
                    if i>-1:
                        pc_stump = CELL(pc_cells.points, pc_cells.intensity)
                        idx_label=np.where(labels_stumps==i)
                        pc_stump.index_cut(idx_label)
                        
                        height = pc_stump.points.max(axis=0)[2]-pc_stump.points.min(axis=0)[2]
                        if height>=cs.height_limit_1:

                            # filename_stumps_out = 'int' + str(intensity_cut_make) + '_' + str(tfni).rjust(4, '0') + '.pcd'
                            # fname_stumps_out = os.path.join(path_file_stumps, 'before_sor', filename_stumps_out) 
                            # pc_stump.save(fname_stumps_out)

                            pc_stump.points, pc_stump.intensity = PCD_UTILS.SOR(pc_stump.points, pc_stump.intensity)

                            # filename_stumps_out = 'int' + str(intensity_cut_make) + '_' + str(tfni).rjust(4, '0') + '.pcd'
                            # fname_stumps_out = os.path.join(path_file_stumps, 'after_sor', filename_stumps_out) 
                            # pc_stump.save(fname_stumps_out)
                            
                            labels_XY = pc_stump.labels_XY_dbscan(eps = cs.eps_XY)

                            for j in np.unique(labels_XY):
                                if j>-1:
                                    pc_stump_clear = CELL(pc_stump.points, pc_stump.intensity)
                                    idx_label=np.where(labels_XY==j)
                                    pc_stump_clear.index_cut(idx_label)

                                    height = pc_stump_clear.points.max(axis=0)[2]-pc_stump_clear.points.min(axis=0)[2]
                                    if height>=cs.height_limit_2:
                                        labels_Z = pc_stump_clear.label_Z_dbscan(eps = cs.eps_Z)

                                        max_shape = 0
                                        i_max_shape = -1
                                        for k in np.unique(labels_Z):
                                            if k>=-1:
                                                pc_stump_verifiable = PCD(pc_stump_clear.points, pc_stump_clear.intensity)
                                                idx_label=np.where(labels_Z==k)
                                                pc_stump_verifiable.index_cut(idx_label)
                                                if pc_stump_verifiable.points.shape[0]>max_shape:
                                                    max_shape = pc_stump_verifiable.points.shape[0]
                                                    i_max_shape = k

                                        if i_max_shape!=-1:
                                            pc_stump_suitable = PCD(pc_stump_clear.points, pc_stump_clear.intensity)
                                            idx_label=np.where(labels_Z==i_max_shape)
                                            pc_stump_suitable.index_cut(idx_label)
                                            
                                            r_list = []
                                            xy_list = []
                                            save_center = [0,0,0]

                                            x_min, y_min, z_min = pc_stump_suitable.points.min(axis=0)
                                            x_max, y_max, z_max = pc_stump_suitable.points.max(axis=0)
                                            if z_max-z_min>1:

                                                num_layers = 4
                                                layer = (z_max-z_min)/num_layers

                                                for l in range(num_layers):
                                                    pc_stump_suitable_layer = PCD(pc_stump_suitable.points, pc_stump_suitable.intensity)
                                                    idx_layer = np.where((pc_stump_suitable_layer.points[:,2]>=l*layer+z_min)&(pc_stump_suitable_layer.points[:,2]<(l+1)*layer+z_min))
                                                    pc_stump_suitable_layer.index_cut(idx_layer)

                                                    try:
                                                        xc,yc,r,_ = cf.hyper_fit(pc_stump_suitable_layer.points)
                                                    except:
                                                        xc,yc,r,_ = 0,0,0,0
                                                    r_list.append(r)
                                                    xy_list.append([xc,yc])
                                                                                        
                                                xy_list=np.asarray(xy_list)
                                                
                                                r_median = statistics.median(r_list)
                                                x_median = statistics.median(xy_list[:,0])
                                                y_median = statistics.median(xy_list[:,1])
                                                check_x = np.median(pc_stump_suitable.points[:,0])
                                                check_y = np.median(pc_stump_suitable.points[:,1])

                                                x_min, y_min, z_min = pc_stump_suitable.points.min(axis=0)
                                                x_max, y_max, z_max = pc_stump_suitable.points.max(axis=0)
                                                check_r_median = ((x_max - x_min) + (y_max - y_min))/4
                                                if (r_median > 0.65) or (r_median > 2.1*check_r_median) or (r_median == 0.0):
                                                    r_median = check_r_median

                                                dist = math.sqrt((xy_list[0][0] - check_x)**2 + (xy_list[0][1] - check_y)**2)
                                                if dist>0.25:
                                                    dist = math.sqrt((x_median - check_x)**2 + (y_median - check_y)**2)
                                                    if dist>0.25:
                                                        save_center = [check_x,check_y,1]
                                                    else:
                                                        save_center = [x_median,y_median,1]
                                                else:
                                                    save_center = [xy_list[0][0],xy_list[0][1],1]

                                                tfni += 1
                                                filename_stumps_out = 'int' + str(intensity_cut_make) + '_' + str(tfni).rjust(4, '0') + '.pcd'
                                                fname_stumps_out = os.path.join(path_file_stumps, filename_stumps_out) 
                                                pc_stump_suitable.save(fname_stumps_out)

                                                TN.append(filename_stumps_out)
                                                TCX.append(save_center[0])
                                                TCY.append(save_center[1])
                                                TD.append(r_median*2)
                if cs.cut_data_method == 'none':
                    break

        TN=np.asarray(TN)
        TCX=np.asarray(TCX)
        TCY=np.asarray(TCY)
        TD=np.asarray(TD)

        bd = pd.DataFrame({"Name_stump"+'_int' + str(intensity_cut_make): TN,"X": TCX,"Y": TCY,"Diameter"+'_int' + str(intensity_cut_make): TD})
        bd.to_csv(file_name_csv, index = False, sep=';')

        file = open(os.path.join(cs.path_base, "coordinates_paths.txt"), "a")
        file.write("\n"+file_name_csv)
        file.close()

if __name__ == "__main__" :
    cs = CS()
    yml_path = "settings\settings.yaml"
    cs.set(yml_path)
    intensity_cut_make = 7000
    coordinates(intensity_cut_make = intensity_cut_make, cs = cs)