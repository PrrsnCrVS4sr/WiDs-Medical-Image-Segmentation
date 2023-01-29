import numpy as np
from glob import glob
import os
from albumentations.augmentations.transforms import CLAHE

def load_data(path,images_folder,masks_folder,i_format,m_format):
    train_X = sorted(glob(os.path.join(path,"training",images_folder,"*."+i_format)))
    train_Y = sorted(glob(os.path.join(path,"training",masks_folder,"*."+m_format)))
    
    test_X = sorted(glob(os.path.join(path,"test",images_folder,"*."+i_format)))
    test_Y = sorted(glob(os.path.join(path,"test",masks_folder,"*."+m_format)))
    
    return train_X, train_Y, test_X, test_Y

def pre_process(image):
    image = image[:,:,1]
    aug = CLAHE(clip_limit=5.0, tile_grid_size=(8, 8), always_apply=True,p=1.0)
    augmented = aug(image=image)
    return (augmented["image"]/127.5-1)
    