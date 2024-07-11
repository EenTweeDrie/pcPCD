import os
from tqdm import tqdm
from classes.PCD_TREE import PCD_TREE
import settings.seg_settings as ss

def makedirs_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def orbit_gif(path_file):
    for fname in tqdm(os.listdir(path_file)):
        if fname.endswith('.pcd'):
            pc_tree = PCD_TREE()
            pc_tree.open(os.path.join(path_file, fname), verbose = False)
            
            gifname = fname.split(".")[0] + "_orbit.gif"
            gif_folder = os.path.join(path_file, 'gifs')
            makedirs_if_not_exist(gif_folder)
            
            path_gif = os.path.join(gif_folder, gifname)
            pc_tree.visual_gif(path_gif, zoom = 0.3, point_size = 3.0)

if __name__ == '__main__':
    path_file = os.path.join(ss.path_base, ss.step1_folder_name, ss.step2_folder_name, ss.step3_folder_name)
    orbit_gif(path_file)   