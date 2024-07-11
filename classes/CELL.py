import numpy as np
from .PCD import PCD
import pyvista
import sys
import os 
from tqdm import tqdm
import pandas as pd
import hdbscan
from sklearn.cluster import DBSCAN

def toFixed(numObj, digits=0):
    return f"{numObj:.{digits}f}"
    
def loading(part, all):
    animation = [f"[■□□□□□□□□□] {part}/{all}  {toFixed(part/all*100,2)}%",f"[■■□□□□□□□□] {part}/{all}  {toFixed(part/all*100,2)}%", f"[■■■□□□□□□□] {part}/{all}  {toFixed(part/all*100,2)}%", f"[■■■■□□□□□□] {part}/{all}  {toFixed(part/all*100,2)}%", f"[■■■■■□□□□□] {part}/{all}  {toFixed(part/all*100,2)}%", f"[■■■■■■□□□□] {part}/{all}  {toFixed(part/all*100,2)}%", f"[■■■■■■■□□□] {part}/{all}  {toFixed(part/all*100,2)}%", f"[■■■■■■■■□□] {part}/{all}  {toFixed(part/all*100,2)}%", f"[■■■■■■■■■□] {part}/{all}  {toFixed(part/all*100,2)}%", f"[■■■■■■■■■■] {part}/{all}  {toFixed(part/all*100,2)}%"]
    sys.stdout.write("\r" + animation[int((part/all*100)//100) % len(animation)])
    sys.stdout.flush()

class CELL(PCD):
    def __init__(self, points = None, intensity = None, cell_size = None, list_cell = None, points_traj = None, big_cell_points = None, big_cell_intensity = None):
        super().__init__(points, intensity)
        self.cell_size = cell_size
        self.list_cell = list_cell
        self.points_traj = points_traj
        self.big_cell_points = big_cell_points
        self.big_cell_intensity = big_cell_intensity
 
    def make_cell_list(self, min, max, verbose = None):
        self.list_cell = []
        i = min[0]
        j = min[1]
        while i<max[0]:
            i += 2*self.cell_size
            while j<max[1]:
                j += 2*self.cell_size
                self.list_cell.append([i,j,0])
            j = min[1]

        if verbose == True:
            p = pyvista.Plotter(window_size=[1000, 1000])
            pdata = pyvista.PolyData(np.asarray(self.list_cell))
            p.add_mesh(pdata, color='#FF0000')
            pdata = pyvista.PolyData(self.points_traj)
            p.add_mesh(pdata, color='#0000FF')
            p.show()
        
    def micro_cell (self, list_for_consideration, cur_i_list_cell, direction):
        
        x_begin = list_for_consideration[0][0]
        y_begin = list_for_consideration[0][1]

        if direction == "right":
            x_cur = x_begin + 2*self.cell_size
            y_cur = y_begin
        elif direction == "left":
            x_cur = x_begin - 2*self.cell_size
            y_cur = y_begin
        elif direction == "up":
            x_cur = x_begin
            y_cur = y_begin + 2*self.cell_size
        elif direction == "down": 
            x_cur = x_begin
            y_cur = y_begin - 2*self.cell_size        
        else:
            print("Error: Wrong direction")

        if [x_cur,y_cur,1] in self.list_cell:
            pass
        elif [x_cur,y_cur,0] in self.list_cell:
            idx_labels=np.where((self.points_traj[:,0]>x_cur-self.cell_size) & (self.points_traj[:,0]<x_cur+self.cell_size) & (self.points_traj[:,1]>y_cur-self.cell_size) & (self.points_traj[:,1]<y_cur+self.cell_size))
            check_points = self.points_traj[idx_labels]
            if check_points.shape[0]==0:
                idx_labels=np.where((self.points[:,0]>x_cur-self.cell_size) & (self.points[:,0]<x_cur+self.cell_size) & (self.points[:,1]>y_cur-self.cell_size) & (self.points[:,1]<y_cur+self.cell_size))
                cell_points = self.points[idx_labels]
                cell_intensity = self.intensity[idx_labels]
                self.list_cell[self.list_cell.index([x_cur,y_cur,0])][2] = 1
                cur_i_list_cell += 1
                self.big_cell_points = np.vstack((self.big_cell_points, cell_points))
                self.big_cell_intensity = np.hstack((self.big_cell_intensity, cell_intensity))
                list_for_consideration.append([self.list_cell[self.list_cell.index([x_cur,y_cur,1])][0],self.list_cell[self.list_cell.index([x_cur,y_cur,1])][1]])

        return list_for_consideration, cur_i_list_cell

    
    def save_all_cells(self, path_file_save, verbose = None):              
        np_list_cell = np.asarray(self.list_cell)
        all_shape = np_list_cell.shape[0]

        big_cell_i = 0
        
        with tqdm(total=all_shape) as pbar:
            while np_list_cell.shape[0]>0:
                cur_i_list_cell = 0
                x_begin = np_list_cell[0][0]
                y_begin = np_list_cell[0][1]

                self.big_cell_points = np.array([[0,0,0]])
                self.big_cell_intensity = np.array([0])
                list_for_consideration = []
                list_for_consideration.append([x_begin, y_begin])

                idx_labels=np.where((self.points[:,0]>x_begin-self.cell_size) & (self.points[:,0]<x_begin+self.cell_size) & (self.points[:,1]>y_begin-self.cell_size) & (self.points[:,1]<y_begin+self.cell_size))
                cell_points = self.points[idx_labels]
                cell_intensity = self.intensity[idx_labels]
                if [x_begin,y_begin,0] in self.list_cell: 
                    self.list_cell[self.list_cell.index([x_begin,y_begin,0])][2] = 1
                cur_i_list_cell += 1

                idx_labels=np.where((self.points_traj[:,0]>x_begin-self.cell_size) & (self.points_traj[:,0]<x_begin+self.cell_size) & (self.points_traj[:,1]>y_begin-self.cell_size) & (self.points_traj[:,1]<y_begin+self.cell_size))
                check_points = self.points_traj[idx_labels]
                if check_points.shape[0]==0:
                    self.big_cell_points = np.vstack((self.big_cell_points, cell_points))
                    self.big_cell_intensity = np.hstack((self.big_cell_intensity, cell_intensity))

                    while len(list_for_consideration)>0:
                        list_for_consideration, cur_i_list_cell = self.micro_cell(list_for_consideration, cur_i_list_cell, 'right')
                        list_for_consideration, cur_i_list_cell = self.micro_cell(list_for_consideration, cur_i_list_cell, 'left')
                        list_for_consideration, cur_i_list_cell = self.micro_cell(list_for_consideration, cur_i_list_cell, 'up')
                        list_for_consideration, cur_i_list_cell = self.micro_cell(list_for_consideration, cur_i_list_cell, 'down')
                        list_for_consideration.pop(0)

                        np_list_cell_reserv = np.asarray(self.list_cell)
                        idx_labels=np.where(np_list_cell_reserv[:,2]==1)
                        part_list = np_list_cell_reserv[idx_labels]
                        # loading(part_list.shape[0],all_shape)
                        pbar.update(1)

                    self.big_cell_points = np.delete(self.big_cell_points, 0, axis = 0)
                    self.big_cell_intensity = np.delete(self.big_cell_intensity, 0, axis = 0)
                        
                    if verbose == True:

                        np_list_cell_show = np.asarray(self.list_cell)
                        idx_labels=np.where(np_list_cell_show[:,2]==1)
                        np_list_cell_show = np_list_cell_show[idx_labels]

                        if self.big_cell_points.shape[0]>0:
                            p1 = pyvista.Plotter(window_size=[1000, 1000])
                            pdata = pyvista.PolyData(self.big_cell_points)
                            p1.add_mesh(pdata)
                            pdata = pyvista.PolyData(np_list_cell_show)
                            p1.add_mesh(pdata, color='#FF0000')
                            pdata = pyvista.PolyData(self.points_traj)
                            p1.add_mesh(pdata, color='#0000FF')
                            p1.show()
                
                    idx_labels=np.where(np_list_cell_reserv[:,2]==0)
                    np_list_cell = np_list_cell_reserv[idx_labels]

                if cur_i_list_cell<=1:
                    np_list_cell = np.delete(np_list_cell, 0, axis = 0)
                
                if self.big_cell_points.shape[0]>10:
                    big_cell_i += 1
                    filename_out = str(big_cell_i).rjust(3, '0') + '.pcd'
                    file_name_data_out = os.path.join(path_file_save, filename_out) 

                    pc_result = PCD(points = self.big_cell_points, intensity = self.big_cell_intensity)
                    pc_result.save(file_name_data_out)


    def extract_stumps_labels(self):
        P = pd.DataFrame(self.points, columns = ['X','Y','Z'])
        X = np.asarray(P)
        clustering = hdbscan.HDBSCAN(min_samples=50, gen_min_span_tree=True).fit(X)
        labels=clustering.labels_
        return labels
    
    def labels_XY_dbscan(self, eps = 0.04):
        P = pd.DataFrame(self.points[:,0:2], columns = ['X','Y'])
        X = np.asarray(P)
        if self.points.shape[0]<85000:
            clustering = DBSCAN(eps=eps, min_samples=50).fit(X) #0.35 perm = 3.5
            labels=clustering.labels_
        else:
            labels = np.zeros(self.points.shape[0])
        return labels
    
    def label_Z_dbscan(self, eps = 0.35):
        P = pd.DataFrame(self.points[:,2], columns = ['Z'])
        X = np.asarray(P)   
        if self.points.shape[0]<50000:
            clustering = DBSCAN(eps=eps, min_samples=50).fit(X) #0.35 perm = 3.5
            labels=clustering.labels_
        else:
            labels = np.zeros(self.points.shape[0])
        return labels





