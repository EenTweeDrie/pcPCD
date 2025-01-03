import os
from classes.PCD_AREA import PCD_AREA
import numpy as np
from tqdm import tqdm
import pandas as pd
import shapely.geometry as geom
from shapely.ops import unary_union
import math



def makedirs_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Функция для создания полигона из точек
def create_polygons(group):
    points = [geom.Point(x, y) for x, y in zip(group['X_'], group['Y_'])]
    merged = unary_union([point.buffer(math.sqrt(2)*0.1, cap_style = 'round') for point in points])  # Размер пикселя 0.2, радиус 0.1
    return merged



path_base = 'D:/lidar/data'
data = pd.read_csv(os.path.join(path_base, 'PR_KRON.csv'), sep=';')
makedirs_if_not_exist(os.path.join(path_base, 'proj_voxels'))
grouped = data.groupby('NAME_')

create_borders = True
use_borders = True

if create_borders:
    # Обработка и сохранение в CSV
    for name, group in grouped:
        polygons = create_polygons(group)

        if polygons.geom_type == 'Polygon':
            polygons = polygons.simplify(tolerance = 0.01, preserve_topology=True)
            polygon_xy = np.asarray(polygons.exterior.coords.xy)       
            polygon = np.vstack((polygon_xy[0], polygon_xy[1])).T.reshape(-1, 2)

            z = np.zeros(polygon.shape[0])
            dt1 = np.c_[polygon, z]
            dt1 = np.array(dt1, dtype=np.float32)
            filename_border = name + '_0.csv'
            fname_brd = os.path.join(path_base, 'proj_voxels', filename_border) 
            np.savetxt(fname_brd, dt1, delimiter=',', header='x,y,z')
            
        elif polygons.geom_type == 'MultiPolygon':
            j = 0
            for poly_item in polygons.geoms:
                poly_item = poly_item.simplify(tolerance = 0.01, preserve_topology=True)
                polygon_xy = np.asarray(poly_item.exterior.coords.xy)       
                polygon = np.vstack((polygon_xy[0], polygon_xy[1])).T.reshape(-1, 2)
                z = np.zeros(polygon.shape[0])
                dt1 = np.c_[polygon, z]
                dt1 = np.array(dt1, dtype=np.float32)
                filename_border = name
                filename_border = filename_border + f"_{j}.csv"
                fname_brd = os.path.join(path_base, 'proj_voxels', filename_border) 
                np.savetxt(fname_brd, dt1, delimiter=',', header='x,y,z')
                j += 1

if use_borders:
    pth = 'D:/lidar/data/orion/orion_cut.las'
    pc = PCD_AREA()
    pc.open(pth)
    folder_path = os.path.join(path_base, 'proj_voxels')
    makedirs_if_not_exist(os.path.join(folder_path, 'proj'))

    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.csv'):
            parts = filename.split('_')

            fname_brd = os.path.join(folder_path, filename)
            border = np.loadtxt(fname_brd, delimiter=',', dtype=np.float32)
            pc_part = pc.poly_cut(border, returned = 'area')

            fname_part = os.path.join(folder_path, 'proj', filename.partition('.csv')[0].partition('_')[0] + '.pcd') 

            if len(parts) > 1 and parts[1].split('.')[0].isdigit() and int(parts[1].split('.')[0]) > 0:
                pc_part.append(prev_pc_part)
                if pc_part.points.shape[0]>1:
                    pc_part.save(fname_part)
            else:
                if pc_part.points.shape[0]>1:
                    pc_part.save(fname_part)
                prev_pc_part = pc_part






