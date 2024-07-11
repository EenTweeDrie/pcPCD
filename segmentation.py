import os
from segmentation_vor import segmentation_vor
from segmentation_ram import segmentation_ram
from segmentation_clear import segmentation_clear
from seg_after import seg_after
from orbit_gif import orbit_gif
from predict import predict
from parameters import parameters
import settings.seg_settings as ss 


if __name__ == "__main__" :
    print("segmentation_vor")
    # segmentation_vor(make_binding = True)
    
    # print("predict")
    model_name = 'cpl1-1024-rp-s1024-pn2'
    # pth = os.path.join(ss.path_base, ss.step1_folder_name)
    # predict(pth, model_name)

    print("segmentation_ram")
    segmentation_ram()

    # pth = os.path.join(ss.path_base, ss.step1_folder_name)
    # predict(pth, model_name)

    print("segmentation_clear")
    segmentation_clear()

    print("predict")
    pth = os.path.join(ss.path_base, ss.step1_folder_name, ss.step2_folder_name, ss.step3_folder_name)
    predict(pth, model_name)

    print("predict")
    # model_name = 'cpl1-1024-deff-2k-bin-s1024-pn2'
    model_name = 'v5_cpl1-1024-rvc-s1024'
    predict(pth, model_name)

    print("seg_after")
    seg_after(model_name)

    print("parameters")
    path_file = os.path.join(ss.path_base, ss.step1_folder_name, ss.step2_folder_name, ss.step3_folder_name, model_name)
    parameters(path_file)

    print("predict")
    model_name = 'cpl1-1024-rp-s1024-pn2'
    predict(path_file, model_name)

    print("orbit_gif")
    orbit_gif(pth)
