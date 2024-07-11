import os
from segmentation_vor import segmentation_vor
from segmentation_ram import segmentation_ram
from segmentation_clear import segmentation_clear
from seg_after import seg_after
from orbit_gif import orbit_gif
from predict import predict
from parameters import parameters
from settings.seg_settings import SS


if __name__ == "__main__" :
    ss = SS()
    yml_path = "settings\settings.yaml"
    ss.set(yml_path)
    print("segmentation_vor")
    segmentation_vor(ss, make_binding = True)
    
    print("segmentation_ram")
    segmentation_ram(ss)

    print("segmentation_clear")
    segmentation_clear(ss)

    model_name = 'cpl1-1024-rp-s1024-pn2'
    path_file = os.path.join(ss.path_base, ss.step1_folder_name, ss.step2_folder_name, ss.step3_folder_name)
    predict(path_file, model_name)   

    path_file = os.path.join(ss.path_base, ss.step1_folder_name, ss.step2_folder_name, ss.step3_folder_name)
    parameters(ss, path_file)


