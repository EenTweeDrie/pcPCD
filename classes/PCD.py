import torch
import numpy as np
import pprint
from time import time
from .PCD_UTILS import PCD_UTILS
import open3d as o3d
import laspy
import pyvista
from pypcd import pypcd
import h5py
import pandas as pd


class PCD:
    def __init__(self, points=None, intensity=None, rgb=None, index=None, gps_time=None, illuminance=None):
        self.points = points
        self.intensity = intensity
        self.rgb = rgb
        self.index = index
        self.gps_time = gps_time
        self.illuminance = illuminance

    def save(self, file_path, verbose=False):
        def save_pcd(self, file_path, verbose=False):
            """ save .pcd """
            if verbose:
                print(f"Saving file {file_path} ...")
                start = time()
            dt = np.zeros((len(self.points), 8), dtype=np.float32)
            dt[:, :3] = self.points
            if self.rgb is not None:
                rgb = np.uint8(self.rgb)
                dt[:, 3] = pypcd.encode_rgb_for_pcl(rgb)
            dt[:, 4] = self.gps_time if self.gps_time is not None else None
            dt[:, 5] = self.index if self.index is not None else None
            dt[:, 6] = self.intensity if self.intensity is not None else None
            dt[:, 7] = self.illuminance if self.illuminance is not None else None
            md = {'version': .7,
                  'fields': ['x', 'y', 'z', 'rgb', 'GpsTime', 'Original_cloud_index', 'Intensity', 'Illuminance'],
                  'count': [1, 1, 1, 1, 1, 1, 1, 1],
                  'width': len(dt),
                  'height': 1,
                  'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                  'points': len(dt),
                  'type': ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
                  'size': [4, 4, 4, 4, 4, 4, 4, 4],
                  'data': 'binary'}
            pc_data = dt.view(np.dtype([('x', np.float32),
                                        ('y', np.float32),
                                        ('z', np.float32),
                                        ('rgb', np.float32),
                                        ('GpsTime', np.float32),
                                        ('Original_cloud_index', np.float32),
                                        ('Intensity', np.float32),
                                        ('Illuminance', np.float32)])).squeeze()

            new_cloud = pypcd.PointCloud(md, pc_data)
            new_cloud.save_pcd(file_path, 'binary')
            if verbose:
                end = time()-start
                print(f"Time saving: {end:.3f} s")

        def save_las(self, file_path, verbose=False):
            """" save .las """
            if verbose:
                print(f"Saving file {file_path} ...")
                start = time()
            header = laspy.LasHeader(point_format=3, version="1.4")
            header.point_count = len(self.points)
            las = laspy.LasData(header)
            las.add_extra_dim(laspy.ExtraBytesParams(
                name="illuminance", type=np.float32))
            self.points = np.asarray(self.points, dtype=np.float32)
            las.x = self.points[:, 0]
            las.y = self.points[:, 1]
            las.z = self.points[:, 2]
            if self.rgb is not None:
                las.red = self.rgb[:, 0] * 256
                las.green = self.rgb[:, 1] * 256
                las.blue = self.rgb[:, 2] * 256
            if self.intensity is not None:
                las.intensity = self.intensity
            if self.illuminance is not None:
                las.illuminance = self.illuminance
            if self.gps_time is not None:
                las.gps_time = self.gps_time
            if self.index is not None:
                las.point_source_id = self.index
            las.write(file_path)
            if verbose:
                end = time()-start
                print(f"Time saving: {end:.3f} s")

        def save_laz(self, file_path, verbose=False):
            """" save .laz """
            if verbose:
                print(f"Saving file {file_path} ...")
                start = time()
            header = laspy.LasHeader(point_format=3, version="1.4")
            header.point_count = len(self.points)
            las = laspy.LasData(header)
            las.add_extra_dim(laspy.ExtraBytesParams(
                name="illuminance", type=np.float32))
            self.points = np.asarray(self.points, dtype=np.float32)
            las.x = self.points[:, 0]
            las.y = self.points[:, 1]
            las.z = self.points[:, 2]
            if self.rgb is not None:
                las.red = self.rgb[:, 0] * 256
                las.green = self.rgb[:, 1] * 256
                las.blue = self.rgb[:, 2] * 256
            if self.intensity is not None:
                las.intensity = self.intensity
            if self.illuminance is not None:
                las.illuminance = self.illuminance
            if self.gps_time is not None:
                las.gps_time = self.gps_time
            if self.index is not None:
                las.point_source_id = self.index
            las.write(file_path)
            if verbose:
                end = time()-start
                print(f"Time saving: {end:.3f} s")

        def save_csv(self, file_path, verbose=False):
            """" save .csv """
            if verbose:
                print(f"Saving file {file_path} ...")
                start = time()
            data = {}
            if self.points is not None:
                points = np.asarray(self.points)
                data["x"] = points[:, 0]
                data["y"] = points[:, 1]
                data["z"] = points[:, 2]
            if self.intensity is not None:
                data["Intensity"] = self.intensity
            if self.illuminance is not None:
                data["Illuminance"] = self.illuminance
            if self.gps_time is not None:
                data["GpsTime"] = self.gps_time
            if self.index is not None:
                data["Original_cloud_index"] = self.index
            if self.rgb is not None:
                rgb = np.asarray(self.rgb)
                data["red"] = rgb[:, 0]
                data["green"] = rgb[:, 1]
                data["blue"] = rgb[:, 2]
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            if verbose:
                end = time()-start
                print(f"Time saving: {end:.3f} s")

        def save_txt(self, file_path, verbose=False):
            """ save .txt """
            if verbose:
                print(f"Saving file {file_path} ...")
                start = time()
            # Determine the columns to write based on available data
            columns_to_write = []
            if self.points is not None:
                columns_to_write.extend(['X', 'Y', 'Z'])
            if self.intensity is not None:
                columns_to_write.append('Intensity')
            if self.rgb is not None:
                columns_to_write.extend(['R', 'G', 'B'])
            if self.index is not None:
                columns_to_write.append('Original_cloud_index')
            if self.gps_time is not None:
                columns_to_write.append('GpsTime')
            if self.illuminance is not None:
                columns_to_write.append('Illuminance_(PCV)')

            # Write the file
            with open(file_path, 'w') as file:
                # Write the header line
                header_line = '//' + ' '.join(columns_to_write)
                file.write(header_line + '\n')

                # Write the data lines
                num_points = len(self.points) if self.points is not None else 0
                for i in range(num_points):
                    values = []
                    if self.points is not None:
                        values.extend(self.points[i])
                    if self.intensity is not None:
                        values.append(self.intensity[i])
                    if self.rgb is not None:
                        values.extend(self.rgb[i])
                    if self.index is not None:
                        values.append(self.index[i])
                    if self.gps_time is not None:
                        values.append(self.gps_time[i])
                    if self.illuminance is not None:
                        values.append(self.illuminance[i])
                    line = ' '.join(map(str, values))
                    file.write(line + '\n')

            if verbose:
                end = time()-start
                print(f"Time saving: {end:.3f} s")

        def save_h5(self, file_path, verbose=False):
            """Save data to an HDF5 file."""
            if verbose:
                start = time()
                print(f"Saving data to {file_path} ...")

            with h5py.File(file_path, 'w') as h5f:
                if self.points is not None:
                    h5f.create_dataset('points', data=self.points)
                if self.intensity is not None:
                    h5f.create_dataset('Intensity', data=self.intensity)
                if self.rgb is not None:
                    h5f.create_dataset('rgb', data=self.rgb)
                if self.index is not None:
                    h5f.create_dataset('Original_cloud_index', data=self.index)
                if self.gps_time is not None:
                    h5f.create_dataset('GpsTime', data=self.gps_time)
                if self.illuminance is not None:
                    h5f.create_dataset('Illuminance',
                                       data=self.illuminance)

            if verbose:
                end = time() - start
                print(f"Time saving data: {end:.3f} s")

        if file_path.endswith('.pcd'):
            save_pcd(self, file_path, verbose=verbose)
        elif file_path.endswith('.las'):
            save_las(self, file_path, verbose=verbose)
        elif file_path.endswith('.laz'):
            save_laz(self, file_path, verbose=verbose)
        elif file_path.endswith('.laz'):
            save_laz(self, file_path, verbose=verbose)
        elif file_path.endswith('.csv'):
            save_csv(self, file_path, verbose=verbose)
        elif file_path.endswith('.txt'):
            save_txt(self, file_path, verbose=verbose)
        elif file_path.endswith('.h5'):
            save_h5(self, file_path, verbose=verbose)
        else:
            print("invalid format")

    def open(self, file_path, verbose=False):
        def open_pcd(self, file_path, verbose=False):
            """ open .pcd """
            if verbose:
                start = time()
                print(f"Opening file {file_path} ...")
            cloud = pypcd.PointCloud.from_path(file_path)
            data = cloud.pc_data.view(np.float32).reshape(
                cloud.pc_data.shape + (-1,))
            ix = cloud.get_metadata()["fields"].index('x')
            self.points = data[:, ix:ix + 3]
            try:
                ii = cloud.get_metadata()["fields"].index('Intensity')
                self.intensity = np.nan_to_num(np.asarray(data[:, ii]))
            except ValueError:
                self.intensity = None
            try:
                il = cloud.get_metadata()["fields"].index('Illuminance')
                self.illuminance = np.nan_to_num(np.asarray(data[:, il]))
            except ValueError:
                self.illuminance = None
            try:
                ir = cloud.get_metadata()["fields"].index('rgb')
                rgb = pypcd.decode_rgb_from_pcl(data[:, ir])
                self.rgb = np.nan_to_num(rgb)
            except ValueError:
                self.rgb = None
            try:
                ig = cloud.get_metadata()["fields"].index('GpsTime')
                self.gps_time = np.nan_to_num(np.asarray(data[:, ig]))
            except ValueError:
                self.gps_time = None
            try:
                iid = cloud.get_metadata()["fields"].index(
                    'Original_cloud_index')
                self.index = np.nan_to_num(np.asarray(data[:, iid]))
            except ValueError:
                self.index = None
            if verbose:
                end = time()-start
                print(f"Time stacking data: {end:.3f} s")

        def open_h5(self, file_path, verbose=False):
            """ open .h5 """
            if verbose:
                start = time()
                print(f"Opening file {file_path} ...")
            h5f = h5py.File(file_path, 'r')
            try:
                self.points = np.asarray(h5f.get('points'))
            except:
                self.points = None
            try:
                self.intensity = np.asarray(h5f.get('Intensity'))
            except:
                self.intensity = None
            try:
                self.illuminance = np.asarray(h5f.get('Illuminance'))
            except:
                self.illuminance = None
            try:
                self.rgb = np.asarray(h5f.get('rgb'))
            except:
                self.rgb = None
            try:
                self.gps_time = np.asarray(h5f.get('GpsTime'))
            except:
                self.gps_time = None
            try:
                self.index = np.asarray(h5f.get('Original_cloud_index'))
            except:
                self.index = None
            h5f.close()
            if verbose:
                end = time()-start
                print(f"Time stacking data: {end:.3f} s")

        def open_las(self, file_path, verbose=False):
            """ open .las """
            if verbose:
                start = time()
                print(f"Opening file {file_path} ...")
            las = laspy.read(file_path)
            points = np.vstack(
                [las.points.x, las.points.y, las.points.z]).transpose()
            self.points = points
            try:
                self.intensity = las.intensity
            except:
                self.intensity = None  # np.full(points.shape[0], 0)
            try:
                self.illuminance = las.illuminance
            except:
                self.illuminance = None
            try:
                rgb = np.vstack(
                    [las.points.red, las.points.green, las.points.blue]).transpose()
                self.rgb = (rgb // 256).astype(np.uint8)
            except:
                # np.zeros((points.shape[0], 3), dtype=np.int32)
                self.rgb = None
            try:
                self.index = las.point_source_id
            except:
                self.index = None
            try:
                self.gps_time = las.gps_time
            except:
                self.gps_time = None
            if verbose:
                end = time()-start
                print(f"Time stacking data: {end:.3f} s")

        def open_laz(self, file_path, verbose=False):
            """ open .laz """
            if verbose:
                start = time()
                print(f"Opening file {file_path} ...")
            with laspy.open(file_path) as fh:
                las = fh.read()
                points = np.vstack(
                    [las.points.x, las.points.y, las.points.z]).transpose()
                self.points = points
                try:
                    self.intensity = np.nan_to_num(
                        np.asarray(las.intensity, dtype=np.int32))
                except:
                    self.intensity = None  # np.full(points.shape[0], 0)
                try:
                    self.illuminance = np.nan_to_num(
                        np.asarray(las.illuminance, dtype=np.int32))
                except:
                    self.illuminance = None
                try:
                    rgb = np.vstack(
                        [las.points.red, las.points.green, las.points.blue]).transpose()
                    self.rgb = (rgb // 256).astype(np.uint8)
                except:
                    # np.zeros((points.shape[0], 3), dtype=np.int32)
                    self.rgb = None
                try:
                    self.index = np.nan_to_num(np.asarray(
                        las.point_source_id, dtype=np.float16))
                except:
                    self.index = None
                try:
                    self.gps_time = np.nan_to_num(
                        np.asarray(las.gps_time, dtype=np.float16))
                except AttributeError:
                    self.gps_time = None
            if verbose:
                end = time()-start
                print(f"Time stacking data: {end:.3f} s")

        def open_csv(self, file_path, verbose=False):
            """ open .csv """
            if verbose:
                start = time()
                print(f"Opening file {file_path} ...")
            df = pd.read_csv(file_path)
            self.points = df[['x', 'y', 'z']
                             ].values if 'x' in df.columns else None
            self.intensity = df['Intensity'].values if 'Intensity' in df.columns else None
            self.gps_time = df['GpsTime'].values if 'GpsTime' in df.columns else None
            self.index = df['Original_cloud_index'].values if 'Original_cloud_index' in df.columns else None
            self.rgb = df[['red', 'green', 'blue']
                          ].values if 'red' in df.columns else None
            self.illuminance = df['Illuminance'].values if 'Illuminance' in df.columns else None
            if verbose:
                end = time()-start
                print(f"Time stacking data: {end:.3f} s")

        def open_txt(self, file_path, verbose=False):
            """ open .txt """

            if verbose:
                start = time()
                print(f"Opening file {file_path} ...")
            # Read the file
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Read the header line
            header = [
                col.strip('//') for col in lines[0].strip().split() if col.startswith('//')]
            if lines[0].startswith('//'):
                header = [col.strip('//') for col in lines[0].split()]
            else:
                if verbose:
                    print("Header is empty. Using default column names.")
                header = ['X', 'Y', 'Z', 'Intensity',
                          'R', 'G', 'B', 'Original_cloud_index', 'GpsTime', 'Illuminance_(PCV)']
            # Initialize dictionaries to store data
            data = {col: [] for col in header}

            # Read the data lines
            for line in lines[1:]:
                values = line.strip().split()
                for col, value in zip(header, values):
                    data[col].append(float(value))

            # Initialize dictionaries to store data
            data = {col: [] for col in header if not col.startswith('//')}

            # Read the data lines
            for line in lines[1:]:
                values = line.strip().split()
                for col, value in zip(header, values):
                    if col.startswith('//'):
                        continue
                    data[col].append(float(value))

            # Convert lists to numpy arrays for easier manipulation
            for col in data:
                data[col] = np.array(data[col])

            # Assign data to attributes
            if 'X' in data and 'Y' in data and 'Z' in data:
                self.points = np.vstack(
                    (data['X'], data['Y'], data['Z'])).T
            if 'Intensity' in data:
                self.intensity = data['Intensity']
            if 'R' in data and 'G' in data and 'B' in data:
                self.rgb = np.vstack((data['R'], data['G'], data['B'])).T
            if 'Original_cloud_index' in data:
                self.index = data['Original_cloud_index']
            if 'GpsTime' in data:
                self.gps_time = data['GpsTime']
            if 'Illuminance_(PCV)' in data:
                self.illuminance = data['Illuminance_(PCV)']
            if verbose:
                end = time()-start
                print(f"Time stacking data: {end:.3f} s")

        if file_path.endswith(".h5"):
            open_h5(self, file_path, verbose=verbose)
        elif file_path.endswith('.pcd'):
            open_pcd(self, file_path, verbose=verbose)
        elif file_path.endswith('.las'):
            open_las(self, file_path, verbose=verbose)
        elif file_path.endswith('.laz'):
            open_laz(self, file_path, verbose=verbose)
        elif file_path.endswith('.csv'):
            open_csv(self, file_path, verbose=verbose)
        elif file_path.endswith('.txt'):
            open_txt(self, file_path, verbose=verbose)
        else:
            print("invalid format")

    def sample_fps(self, num_sample, verbose=False):
        """ sampling 'num_sample' points from 'PCD' class via farthest point sampling algorithm """
        start = time()
        if verbose:
            end = time() - start
            print(f"Time sampling (fps): {end:.3f} s")
        np_points = np.asarray([self.points])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        points_torch = torch.Tensor(np_points).to(device)
        centroids = PCD_UTILS.farthest_point_sample(points_torch, num_sample)
        pt_sampled = points_torch[0][centroids[0]]
        centroids = centroids.cpu().data.numpy()
        int_sampled = self.intensity[centroids[0]]
        rgb_sampled = self.rgb[centroids[0]]
        index_sampled = self.index[centroids[0]]
        gps_time_sampled = self.gps_time[centroids[0]]
        illuminance_sampled = self.illuminance[centroids[0]]
        pt_sampled = pt_sampled.cpu().detach().numpy()
        self.points, self.intensity, self.rgb, self.index, self.gps_time, self.illuminance = pt_sampled, int_sampled, rgb_sampled, index_sampled, gps_time_sampled, illuminance_sampled

    def index_cut(self, idx_labels):
        """ cut points and intensity using indexes """
        self.points = self.points[idx_labels]
        try:
            self.intensity = self.intensity[idx_labels]
        except:
            self.intensity = None
        try:
            self.index = self.index[idx_labels]
        except:
            self.index = None
        try:
            self.gps_time = self.gps_time[idx_labels]
        except:
            self.gps_time = None
        try:
            self.rgb = self.rgb[idx_labels]
        except:
            self.rgb = None
        try:
            self.illuminance = self.illuminance[idx_labels]
        except:
            self.illuminance = None

    def get_normals(self):
        """ return normals """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        normals = np.asarray(pcd.normals)
        return normals

    def unique(self):
        """ leaves only unique point values """
        self.points, unique_indices = np.unique(
            self.points, axis=0, return_index=True)
        if self.intensity is not None:
            self.intensity = np.take(self.intensity, unique_indices)
        if self.rgb is not None:
            self.rgb = np.take(self.rgb, unique_indices, axis=0)
        if self.index is not None:
            self.index = np.take(self.index, unique_indices)
        if self.gps_time is not None:   
            self.gps_time = np.take(self.gps_time, unique_indices)
        if self.illuminance is not None:
            self.illuminance = np.take(self.illuminance, unique_indices)

    def concatenate(self, data):
        dt = np.c_[self.points, self.intensity,
                   self.rgb, self.index, self.gps_time]
        dt = np.concatenate((dt, data), axis=0)
        dt = np.array(dt, dtype=np.float32)
        self.points = dt[:, 0:3]
        self.intensity = dt[:, 3]
        self.rgb = dt[:, 4:7]
        self.index = dt[:, 7]
        self.gps_time = dt[:, 8]
        self.illuminance = dt[:, 9]

    def append(self, other):
        if not isinstance(other, PCD):
            raise TypeError("Argument must be an instance of PCD")
        self.points = np.concatenate((self.points, other.points), axis=0)
        self.intensity = np.concatenate(
            (self.intensity, other.intensity), axis=0)
        self.rgb = np.concatenate((self.rgb, other.rgb), axis=0)
        self.index = np.concatenate((self.index, other.index), axis=0)
        self.gps_time = np.concatenate((self.gps_time, other.gps_time), axis=0)
        self.illuminance = np.concatenate(
            (self.illuminance, other.illuminance), axis=0)

    def visual_gif(self, path_gif, zoom=0.4, point_size=4.0):
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
        path = pl.generate_orbital_path(
            n_points=36, shift=cloud.length/3, factor=3.0)
        pl.open_gif(path_gif)
        pl.orbit_on_path(path, write_frames=True)
        pl.close()
