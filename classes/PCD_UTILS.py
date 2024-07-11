import torch
import numpy as np
import pprint
from time import time
from pypcd import pypcd #python -m pip install git+https://github.com/DanielPollithy/pypcd.git   
import open3d as o3d
import pyvista
import random
import shapefile
from shapely.geometry import Point, Polygon
from numba import jit, njit
import numba

class PCD_UTILS:
    def make_xyz_intensity_point_cloud(xyz_intensity, metadata=None):
        """ Make a pointcloud object from xyz array.
        xyz array is assumed to be float32.
        intensity is assumed to be encoded as float32 according to pcl conventions.
        """
        md = {'version': .7,
            'fields': ['x', 'y', 'z', 'Intensity'],
            'count': [1, 1, 1, 1],
            'width': len(xyz_intensity),
            'height': 1,
            'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            'points': len(xyz_intensity),
            'type': ['F', 'F', 'F', 'F'],
            'size': [4, 4, 4, 4],
            'data': 'binary'}
        if xyz_intensity.dtype != np.float32:
            raise ValueError('array must be float32')
        if metadata is not None:
            md.update(metadata)
        pc_data = xyz_intensity.view(np.dtype([('x', np.float32),
                                        ('y', np.float32),
                                        ('z', np.float32),
                                        ('Intensity', np.float32)])).squeeze()
        pc = pypcd.PointCloud(md, pc_data)
        return pc


    def PCD_OPEN_X_INT_RGB(file_path, verbose = False):
        """ Return data, indexes of fields 'x', 'Intensity' and 'rgb'
        Input:
            file_path: string, /path/to/file/example.foo
            verbose: boolean, enable print info
        Return:
            new_cloud_data: data from file
            ix, ii, ir = integer, indexes of fields 'x', 'Intensity' and 'rgb'
        """
        start = time()
        cloud = pypcd.PointCloud.from_path(file_path)
        new_cloud_data = cloud.pc_data.copy()
        if verbose:
            print(f"Opening {file_path}")
            pprint.pprint(cloud.get_metadata())
        new_cloud_data = cloud.pc_data.view(np.float32).reshape(cloud.pc_data.shape + (-1,))
        if verbose:
            print(new_cloud_data)
            print(f"Shape: {new_cloud_data.shape}")
            end = time()-start
            print(f"Time opening: {end:.3f} s")
        try:
            ii = cloud.get_metadata()["fields"].index('Intensity')
        except ValueError:
            ii = None
        try:
            ir = cloud.get_metadata()["fields"].index('rgb')
        except ValueError:
            ir = None
        ix = cloud.get_metadata()["fields"].index('x')
        return new_cloud_data, ix, ii, ir 
    
    
    def farthest_point_sample(xyz, npoint, verbose = False):
        """
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        device = xyz.device
        batchsize, ndataset, dimension = xyz.shape
        centroids = torch.zeros(batchsize, npoint, dtype=torch.long).to(device)
        distance = torch.ones(batchsize, ndataset).to(device) * 1e10
        farthest =  torch.randint(0, ndataset, (batchsize,), dtype=torch.long).to(device)
        batch_indices = torch.arange(batchsize, dtype=torch.long).to(device)
        for i in range(npoint):
            centroids[:,i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(batchsize, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids
    
    
    def voronoi_finite_polygons_2d(vor, radius=None):
        """
        Reconstruct infinite voronoi regions in a 2D diagram to finite
        regions.
        Parameters
        ----------
        vor : Voronoi
            Input diagram
        radius : float, optional
            Distance to 'points at infinity'.
        Returns
        -------
        regions : list of tuples
            Indices of vertices in each revised Voronoi regions.
        vertices : list of tuples
            Coordinates for revised Voronoi vertices. Same as coordinates
            of input vertices, with 'points at infinity' appended to the
            end.
        """

        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        if radius is None:
            radius = vor.points.ptp().max()*2

        # Construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]

            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue

            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue

                # Compute the missing endpoint of an infinite ridge

                t = vor.points[p2] - vor.points[p1] # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            # finish
            new_regions.append(new_region.tolist())

        return new_regions, np.asarray(new_vertices)

    def inPolygon(x, y, xp, yp):
        """ Return true if x,y in tuples xp, yp
        Input:
            x, y : coordinates of point
            xp, yp : tuple, coordinates of polygons
        Return:
            c: boolean, true if point is inside the polygon else false

        >>> inPolygon(100, 0, (-100, 100, 100, -100), (100, 100, -100, -100))
        """
        c=0
        for i in range(len(xp)):
            if (((yp[i]<=y and y<yp[i-1]) or (yp[i-1]<=y and y<yp[i])) and 
                (x > (xp[i-1] - xp[i]) * (y - yp[i]) / (yp[i-1] - yp[i]) + xp[i])): c = 1 - c    
        return c

    def is_point_inside_polygon(polygon, point):
        polygon_obj = Polygon(polygon)
        point_obj = Point(point)
        return polygon_obj.contains(point_obj)
    
    def move_polygon(polygon, offsetX, offsetY, main_center, pump):
        """ Shifts the coordinates of the polygon, taking into account the offset and a small blow-up
        Input:
            polygon : coordinates of polygon
            offsetX, offsetY : integer, offsets
            main_center : center for blow-up calculation
        Return:
            new_polygon: new coordinates of polygon
        """
        polyX, polyY = polygon[:,0], polygon[:,1]
        
        # offset calculation
        polyX = [x + offsetX for x in polyX]
        polyY = [y + offsetY for y in polyY]
        
        # blow-up calculation
        xi = 0
        yi = 0
        for x in polyX:
            if x>main_center[0]:
                polyX[xi] = polyX[xi] + 0.25*offsetX
            else:
                polyX[xi] = polyX[xi] - 0.25*offsetX
            xi += 1
        for y in polyY:
            if y>main_center[1]:
                polyY[yi] = polyY[yi] + 0.25*offsetX
            else:
                polyY[yi] = polyY[yi] - 0.25*offsetX
            yi += 1  

        # constant blow-up calculation (pump)
        xi = 0
        yi = 0
        for x in polyX:
            if x>main_center[0]:
                polyX[xi] = polyX[xi] + pump
            else:
                polyX[xi] = polyX[xi] - pump
            xi += 1
        for y in polyY:
            if y>main_center[1]:
                polyY[yi] = polyY[yi] + pump
            else:
                polyY[yi] = polyY[yi] - pump
            yi += 1  
        
        new_polygon = np.c_[polyX, polyY]
              
        return new_polygon

    def center_m(coords):
        """ Return barycenter of points based on median """
        x, y = None, None
        if isinstance(coords, np.ndarray):
            x = coords[:, 0]
            y = coords[:, 1]
        elif isinstance(coords, list):
            x = np.array([point[0] for point in coords])
            y = np.array([point[1] for point in coords])
        else:
            raise Exception("Parameter 'coords' is an unsupported type: " + str(type(coords)))
        # coordinates of the barycenter
        x_m = np.median(x)
        y_m = np.median(y)
        center = x_m, y_m
        return center
    
    def SOR (points, intensity = None):
        """ statistical outliers remove algorithm """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        if intensity is not None:
            res_intensity = intensity[ind]   

        inlier_cloud = pcd.select_by_index(ind)
        res_points = np.asarray(inlier_cloud.points)
        if intensity is not None:
            return res_points, res_intensity 
        else:
            return res_points
    
    def visual(points):
        p = pyvista.Plotter(window_size=[1000, 1000])
        pdata = pyvista.PolyData(points)
        sphere = pyvista.Sphere(radius=0.02, phi_resolution=10, theta_resolution=10)
        pc = pdata.glyph(scale=False, geom=sphere, orient=False)
        p.add_mesh(pc)
        p.show()

    def visual_many(p, points, i, labels, main_cluster_id):
        idx_layer=np.where(labels==i)
        i_data = points[idx_layer]
        pdata = pyvista.PolyData(i_data)
        sphere = pyvista.Sphere(radius=0.02, phi_resolution=10, theta_resolution=10)
        pc = pdata.glyph(scale=False, geom=sphere, orient=False)
        if i == main_cluster_id:
            color = [255,0,0]
        elif i == -1:
            color = [0,0,0]
        else:
            r = lambda: random.randint(0,200)
            color = [r(),r(),r()]
        p.add_mesh(pc, color)
        return p
    
    def shp_open(file_shape):
        shape = shapefile.Reader(file_shape)
        feature = shape.shapeRecords()[0]
        first = feature.shape.__geo_interface__  
        result = np.asarray(first['coordinates'][0])
        if result.shape[0] == 2:
            result = np.asarray(first['coordinates'])
        return result
    
    def shift(points, x, y, z):
        x_shift = np.full((points.shape[0],1), x)
        y_shift = np.full((points.shape[0],1), y)
        z_shift = np.full((points.shape[0],1), z)
        if points.shape[1] == 3:
            shift_matrix = np.concatenate([x_shift, y_shift, z_shift], axis=1)
        elif points.shape[1] == 2:
            shift_matrix = np.concatenate([x_shift, y_shift], axis=1)
        elif points.shape[1] == 1:
            shift_matrix = np.concatenate([x_shift], axis=1)
        return points + shift_matrix
    
    def shp_create(pc):
        min_values = np.min(pc.points, axis=0)
        max_values = np.max(pc.points, axis=0)
        shp_poly = np.array([
            [min_values[0], min_values[1]],
            [max_values[0], min_values[1]],
            [max_values[0], max_values[1]],
            [min_values[0], max_values[1]]
        ])
        return shp_poly
    
    def toFixed(numObj, digits=0):
        return f"{numObj:.{digits}f}"

    
    



