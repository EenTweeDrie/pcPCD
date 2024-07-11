import yaml

class CS():
    def __init__(self, FLAG_cut_data = None, FLAG_make_cells = None, FLAG_make_stumps = None, cut_data_method = None, LOW = None, UP = None, x_shift = None, y_shift = None, z_shift = None, algo = None, n_clusters = None, cell_size = None, height_limit_1 = None, height_limit_2 = None, eps_XY = None, eps_Z = None, path_base = None, fname_points = None, fname_traj = None, fname_shape = None):
        self.FLAG_cut_data = FLAG_cut_data
        self.FLAG_make_cells = FLAG_make_cells
        self.FLAG_make_stumps = FLAG_make_stumps
        self.cut_data_method = cut_data_method
        self.LOW = LOW
        self.UP = UP
        self.x_shift = x_shift
        self.y_shift = y_shift
        self.z_shift = z_shift
        self.algo = algo
        self.n_clusters = n_clusters
        self.intensity_cut_vor_tes = 20000
        self.intensity_cut = 0
        self.cell_size = cell_size
        self.height_limit_1 = height_limit_1
        self.height_limit_2 = height_limit_2
        self.eps_XY = eps_XY
        self.eps_Z = eps_Z
        self.path_base = path_base
        self.fname_points = fname_points
        self.fname_traj = fname_traj
        self.fname_shape = fname_shape

    def set(self, yml_path):
        with open(yml_path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        self.FLAG_cut_data = data['FLAG_cut_data']
        self.FLAG_make_cells = data['FLAG_make_cells']
        self.FLAG_make_stumps = data['FLAG_make_stumps']
        self.cut_data_method = data['cut_data_method']
        self.LOW = data['LOW']
        self.UP = data['UP']
        self.x_shift = data['x_shift']
        self.y_shift = data['y_shift']
        self.z_shift = data['z_shift']
        self.algo = data['algo']
        self.n_clusters = data['n_clusters']
        self.intensity_cut_vor_tes = data['intensity_cut_vor_tes']
        self.intensity_cut = data['intensity_cut']
        self.cell_size = data['cell_size']
        self.height_limit_1 = data['height_limit_1']
        self.height_limit_2 = data['height_limit_2']
        self.eps_XY = data['eps_XY']
        self.eps_Z = data['eps_Z']
        self.path_base = data['path_base']
        self.fname_points = data['fname_points']
        self.fname_traj = data['fname_traj']
        self.fname_shape = data['fname_shape']