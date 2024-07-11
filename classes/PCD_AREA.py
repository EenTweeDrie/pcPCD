from .PCD import PCD
from .PCD_UTILS import PCD_UTILS
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from .PCD_TREE import PCD_TREE
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from tqdm import tqdm
from .is_inside import is_inside_sm_parallel, parallelpointinpolygon, ray_tracing_numpy_numba, is_inside_postgis_parallel

class PCD_AREA(PCD):
    def __init__(self, points = None, intensity = None, coordinates = None, polygons = None, shp_poly = None):
        super().__init__(points, intensity)
        self.coordinates = coordinates
        self.polygons = polygons
        self.shp_ply = Polygon(shp_poly)

            
    def vor_regions(self, verbose = False):
        arr_x1_well = np.array(self.coordinates[:, 0], dtype='float')
        arr_y1_well = np.array(self.coordinates[:, 1], dtype='float')
        vor_points = np.array([arr_x1_well, arr_y1_well]).transpose()
        vor = Voronoi(vor_points)
        regions, vertices = PCD_UTILS.voronoi_finite_polygons_2d(vor)
        polygons = []
        for region in regions:
            poly = Polygon(vertices[region])
            intersct = poly.intersection(self.shp_ply)
            xintersct, yintersct = intersct.exterior.xy
            polygons.append(np.asarray(list(zip(xintersct, yintersct))))
        self.polygons = polygons

        if verbose:
            # fig = voronoi_plot_2d(vor, show_vertices=0)
            # plt.show()

            for region in regions:
                polygon = vertices[region]
                plt.fill(*zip(*polygon), alpha=0.4)

            plt.plot(vor_points[:,0], vor_points[:,1], 'ko')
            plt.axis('equal')
            plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
            plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)
            plt.show()
   
    def poly_cut(self, polygon, mode = 'current', returned = 'tree', algo = 'cm_parallel'):
        idx_labels=np.where((self.points[:,0]>min(polygon[:,0])) & (self.points[:,0]<max(polygon[:,0])) & (self.points[:,1]>min(polygon[:,1])) & (self.points[:,1]<max(polygon[:,1])))
        pc_poly = PCD_TREE(self.points, self.intensity)
        if returned == 'area':
            pc_poly = PCD_AREA(self.points, self.intensity)
            pc_poly.coordinates = self.coordinates
            pc_poly.shp_ply = self.shp_ply
        pc_poly.index_cut(idx_labels)

        if algo == 'within':
            j=0
            if returned == 'area':
                pcpoints = tqdm(pc_poly.points)
            else:
                pcpoints = pc_poly.points
            polygon_obj = Polygon(polygon)
            false_array = np.full((pc_poly.points.shape[0],), False)
            for cp in pcpoints:
                cpp = Point(cp[0], cp[1])
                if cpp.within(polygon_obj):
                    false_array[j] = True
                j+=1
            idx_labels = np.where(false_array)

        if algo == 'cm_parallel':
            idx_labels = is_inside_sm_parallel(pc_poly.points,polygon)
        
        if algo == 'inpoly_parallel':
            idx_labels = parallelpointinpolygon(pc_poly.points, polygon)

        if algo == 'ray_tracing':
            idx_labels = ray_tracing_numpy_numba(pc_poly.points, polygon)
            
        if algo == 'postgis_parallel':
            idx_labels = is_inside_postgis_parallel(pc_poly.points, polygon)
        
        pc_poly.points = pc_poly.points[idx_labels]
        pc_poly.intensity = pc_poly.intensity[idx_labels]


        if (mode == 'main') & (returned == 'tree'):
            pc_poly.polygon = polygon
            idx_labels_p=np.where((self.coordinates[:,0]>min(polygon[:,0])) & (self.coordinates[:,0]<max(polygon[:,0])) & (self.coordinates[:,1]>min(polygon[:,1])) & (self.coordinates[:,1]<max(polygon[:,1])))
            for pt in self.coordinates[idx_labels_p]:
                if PCD_UTILS.inPolygon(pt[0], pt[1], tuple(polygon[:,0]), tuple(polygon[:,1]))==1:
                    pc_poly.coordinate = [pt[0],pt[1],0]
        elif returned == 'tree':
            pc_poly.coordinate = None
        
        return pc_poly

    def make_layer_polygon(self, polygon, offsetX, offsetY, main_center, LOW, HIGH):
        
        pump = 0
        polygon = PCD_UTILS.move_polygon(polygon, offsetX, offsetY, main_center, pump)  

        pc_layer = PCD_AREA(self.points, self.intensity)
        idx_labels=np.where((pc_layer.points[:,2]>LOW)&(pc_layer.points[:,2]<=HIGH))
        pc_layer.index_cut(idx_labels)
        pc_layer_polygon = pc_layer.poly_cut(polygon)

        return pc_layer_polygon
    
   

        
