import yaml

class SS():
    def __init__(self, path_base = None, fname_points = None, fname_shape = None, csv_name_coord = None, first_num = None, STEP = None, z_thresholds = None, eps_steps = None, min_pts = None):
        self.path_base = path_base
        self.fname_points = fname_points
        self.fname_shape = fname_shape
        self.csv_name_coord = csv_name_coord
        self.first_num = first_num
        self.STEP = STEP
        self.z_thresholds = z_thresholds
        self.eps_steps = eps_steps
        self.min_pts = min_pts
        self.step1_folder_name = 'vor'
        self.step2_folder_name = 'ram'
        self.step3_folder_name = 'clear'

    def set(self, yml_path):
        with open(yml_path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        self.path_base = data['path_base']
        self.fname_points = data['fname_points']
        self.fname_shape = data['fname_shape']
        self.csv_name_coord = data['csv_name_coord']
        self.first_num = data['first_num']
        self.STEP = data['STEP']
        self.z_thresholds = data['z_thresholds']
        self.eps_steps = data['eps_steps']
        self.min_pts = data['min_pts']