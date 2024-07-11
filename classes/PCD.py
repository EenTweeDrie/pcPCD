import torch
import numpy as np
import pprint
from time import time
from .PCD_UTILS import PCD_UTILS
import open3d as o3d
import laspy
import pyvista

class PCD:
    def __init__(self, points = None, intensity = None):
        self.points = points
        self.intensity = intensity
 
    def save(self, file_path, verbose = False):
        """ save .pcd with intensity """
        if verbose:
            print(f"Saving .pcd file ...")
        start = time()
        self.points = np.asarray(self.points)
        if self.intensity is None:
            self.intensity = np.full(self.points.shape[0], 0)
        dt = np.c_[self.points, self.intensity]
        dt = np.array(dt, dtype=np.float32)
        new_cloud = PCD_UTILS.make_xyz_intensity_point_cloud(dt)
        if verbose:
            pprint.pprint(new_cloud.get_metadata())
            print(dt, dt.shape)
        new_cloud.save_pcd(file_path, 'binary')
        if verbose:
            end = time()-start
            print(f"Time saving: {end:.3f} s")


    def open(self, file_path, mode = 'intensity', verbose = False):
        """ open .pcd with intensity """
        if file_path.endswith('.pcd'):
            if verbose:
                start = time()
                print(f"Opening .pcd file ...")
            data, ix, ii, ir = PCD_UTILS.PCD_OPEN_X_INT_RGB(file_path, verbose)
            if verbose:
                end = time()-start
                print(f"Time reading: {end:.3f} s")
                start = time()
            points = data[:,ix:ix + 3]
            if mode == 'rgb':
                intensity = np.asarray(data[:,ir]) if ir is not None else None
            else:
                intensity = np.asarray(data[:,ii]) if ii is not None else None
            intensity = np.nan_to_num(intensity)
            if verbose:
                end = time()-start
                print(f"Time stacking data: {end:.3f} s")
            self.points, self.intensity = points, intensity

        if file_path.endswith('.las'):
            if verbose:
                start = time()
                print(f"Opening .las file ...")
            las = laspy.read(file_path)
            if verbose:
                end = time()-start
                print(f"Time reading: {end:.3f} s")
                start = time()
            points = np.vstack([las.points.x, las.points.y, las.points.z]).transpose()
            intensity = np.asarray(las.intensity, dtype = np.int32)
            intensity = np.nan_to_num(intensity)
            if verbose:
                end = time()-start
                print(f"Time stacking data: {end:.3f} s")
            self.points, self.intensity = points, intensity

    def sample_fps(self, num_sample, verbose = False):
        """ sampling 'num_sample' points from 'PCD' class via farthest point sampling algorithm """
        start = time()
        if verbose:
            end = time()-start
            print(f"Time sampling (fps): {end:.3f} s")
        np_points = np.asarray([self.points])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        points_torch = torch.Tensor(np_points).to(device)
        centroids = PCD_UTILS.farthest_point_sample(points_torch, num_sample)
        pt_sampled = points_torch[0][centroids[0]]
        centroids = centroids.cpu().data.numpy()
        int_sampled = self.intensity[centroids[0]]
        pt_sampled = pt_sampled.cpu().detach().numpy()
        self.points, self.intensity = pt_sampled, int_sampled

    def index_cut(self, idx_labels):
        """ cut points and intensity using indexes """
        self.points = self.points[idx_labels]
        self.intensity = self.intensity[idx_labels]

    def get_normals(self):
        """ return normals """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        normals = np.asarray(pcd.normals)
        return normals
    
    def unique(self):
        """ leaves only unique point values """
        self.points, unique_indices = np.unique(self.points, axis=0, return_index=True)
        self.intensity = np.take(self.intensity, unique_indices)
    
    def concatenate(self, data):
        dt = np.c_[self.points, self.intensity]
        dt = np.concatenate((dt, data), axis=0)
        dt = np.array(dt, dtype=np.float32)
        self.points = dt[:,0:3]
        self.intensity = dt[:,3]

    def visual_gif(self, path_gif, zoom = 0.4, point_size = 4.0):
        cloud = pyvista.PointSet(self.points)
        scalars = np.linalg.norm(cloud.points - cloud.center, axis=1)
        pl = pyvista.Plotter(off_screen=True)
        pl.add_mesh(
            cloud,
            color='#fff7c2',
            scalars=scalars,
            opacity=1,
            point_size=point_size,
            show_scalar_bar=False,
        )
        pl.background_color = 'k'
        pl.show(auto_close=False)
        pl.camera.zoom(zoom)
        path = pl.generate_orbital_path(n_points=36, shift=cloud.length/3, factor=3.0)
        pl.open_gif(path_gif)
        pl.orbit_on_path(path, write_frames=True)
        pl.close()


