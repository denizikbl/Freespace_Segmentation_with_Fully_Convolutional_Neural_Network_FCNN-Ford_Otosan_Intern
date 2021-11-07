import torch
from matplotlib import pyplot as plt
from Model_uNET import FoInternNet
from preprocess import tensorize_image, tensorize_mask, image_mask_check
import os
import glob
import numpy as np
import torch.nn as nn
import torch.optim as optim
import tqdm
import cv2
import matplotlib.ticker as mticker
from random import shuffle

input_shape = (224,224)
cuda = True
IMAGE_DIR = os.path.join('../data/p1_test/img')
image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()
#print(image_path_list)
#aug_image_list_full = glob.glob(os.path.join(AUG_IMAGE_DIR, "*"))
#aug_image_list_full.sort()

MASK_DIR = os.path.join('../data/p1_test/masks')
mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()
#print(mask_path_list)

model = torch.load(r"D:\GitHub\intern-p1\data\models\lastwadam_model.pth")
model.eval()

if cuda:
    model = model.cuda()


test_input_path_list = image_path_list
test_label_path_list = mask_path_list

def predict(test_input_path_list):
    
    for i in tqdm.tqdm(range(len(test_input_path_list))):
        batch_test = test_input_path_list[i:i+1]
        test_input = tensorize_image(batch_test, input_shape, cuda)
        outs = model(test_input)
        out=torch.argmax(outs,axis=1)
        out_cpu = out.cpu()
        outputs_list=out_cpu.detach().numpy()
        mask=np.squeeze(outputs_list,axis=0)
            
        img=cv2.imread(batch_test[0])
        #mg=cv2.resize(img,(224,224))
        mask = cv2.resize(mask.astype(np.uint8), (1920,1208))
        mask_ind   = mask == 1
        cpy_img  = img.copy()
        img[mask==1 ,:] = (255, 0, 125)
        opac_image=(img/2+cpy_img/2).astype(np.uint8)
        predict_name=batch_test[0]
        predict_path= predict_name.replace('img', 'predict')
        
        cv2.imwrite(predict_path,opac_image.astype(np.uint8))

predict(test_input_path_list)
