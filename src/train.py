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

######### PARAMETERS ##########
valid_size = 0.3
test_size  = 0.1
batch_size = 4
epochs = 30
cuda = True
augmentation = True
input_shape = (224, 224)
n_classes = 2
###############################

######### DIRECTORIES #########
SRC_DIR = os.getcwd()
ROOT_DIR = os.path.join(SRC_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
MASK_DIR = os.path.join(DATA_DIR, 'masks')
AUG_IMAGE_DIR=os.path.join(DATA_DIR,'AUG_IMAGE_DIR')
AUG_MASK_DIR=os.path.join(DATA_DIR,'AUG_MASK_DIR')
###############################


# PREPARE IMAGE AND MASK LISTS
image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()

aug_image_list_full = glob.glob(os.path.join(AUG_IMAGE_DIR, "*"))
aug_image_list_full.sort()

mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()

aug_mask_list_full = glob.glob(os.path.join(AUG_MASK_DIR, "*"))
aug_mask_list_full.sort()

image_path_list = image_path_list + aug_image_list_full
mask_path_list = mask_path_list + aug_mask_list_full

N = len(image_path_list)
ind_list = [i for i in range(N)]
shuffle(ind_list)

image_path_list = list(np.array(image_path_list)[ind_list])
mask_path_list = list(np.array(mask_path_list)[ind_list])


# DATA CHECK
image_mask_check(image_path_list, mask_path_list)

# SHUFFLE INDICES
indices = np.random.permutation(len(image_path_list))

# DEFINE TEST AND VALID INDICES
test_ind  = int(len(indices) * test_size)
valid_ind = int(test_ind + len(indices) * valid_size)

# SLICE TEST DATASET FROM THE WHOLE DATASET
test_input_path_list = image_path_list[:test_ind]
test_label_path_list = mask_path_list[:test_ind]

# SLICE VALID DATASET FROM THE WHOLE DATASET
valid_input_path_list = image_path_list[test_ind:valid_ind]
valid_label_path_list = mask_path_list[test_ind:valid_ind]

# SLICE TRAIN DATASET FROM THE WHOLE DATASET
train_input_path_list = image_path_list[valid_ind:]
train_label_path_list = mask_path_list[valid_ind:]

# DEFINE STEPS PER EPOCH
steps_per_epoch = len(train_input_path_list)//batch_size

# CALL MODEL
model = FoInternNet(input_size=input_shape, n_classes=2)

# DEFINE LOSS FUNCTION AND OPTIMIZER
criterion = nn.BCELoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=False)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)


if augmentation:
    # PREPARE AUGMENTATED IMAGE AND MASK LISTS
    
    aug_image_list_full = glob.glob(os.path.join(AUG_IMAGE_DIR, "*"))
    aug_image_list_full.sort()
    aug_image_list = aug_image_list_full[valid_ind:]
    
    aug_mask_list_full = glob.glob(os.path.join(AUG_MASK_DIR, "*"))
    aug_mask_list_full.sort()
    aug_mask_list = aug_mask_list_full[valid_ind:]
    
    aug_size = int(len(aug_mask_list)/2)
    
    train_input_path_list = aug_image_list[:aug_size] + train_input_path_list + aug_image_list[aug_size:]
    train_label_path_list = aug_mask_list[:aug_size] + train_label_path_list + aug_mask_list[aug_size:]


# IF CUDA IS USED, IMPORT THE MODEL INTO CUDA
if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

val_losses=[]
train_losses=[]
# TRAINING THE NEURAL NETWORK
for epoch in range(epochs):
    running_loss = 0
    for ind in tqdm.tqdm(range(steps_per_epoch)):
        #print(test_input_path_list)
        batch_input_path_list = train_input_path_list[batch_size*ind:batch_size*(ind+1)]
        batch_label_path_list = train_label_path_list[batch_size*ind:batch_size*(ind+1)]
        batch_input = tensorize_image(batch_input_path_list, input_shape, cuda)
        batch_label = tensorize_mask(batch_label_path_list, input_shape, n_classes, cuda)

        optimizer.zero_grad()
        model.zero_grad()
        #print(batch_input.shape)
        
        outputs = model(batch_input)
        
        loss = criterion(outputs, batch_label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        #print(ind)
        if ind == steps_per_epoch-1:
            train_losses.append(running_loss)
            print('training loss on epoch {}: {}'.format(epoch, running_loss))
            val_loss = 0
            for (valid_input_path, valid_label_path) in zip(valid_input_path_list, valid_label_path_list):
                batch_input = tensorize_image([valid_input_path], input_shape, cuda)
                batch_label = tensorize_mask([valid_label_path], input_shape, n_classes, cuda)
                outputs = model(batch_input)
                loss = criterion(outputs, batch_label)
                val_loss += loss.item()
                val_losses.append(val_loss)
                break

            print('validation loss on epoch {}: {}'.format(epoch, val_loss))

torch.save(model, r'D:\GitHub\intern-p1\data\models/last_model.pth')
print("Model Saved!")
last_model = torch.load(r'D:\GitHub\intern-p1\data\models/last_model.pth')    

def draw_graph(val_losses,train_losses,epochs):
    norm_validation = [float(i)/sum(val_losses) for i in val_losses]
    norm_train = [float(i)/sum(train_losses) for i in train_losses]
    epoch_numbers=list(range(1,epochs+1,1))
    plt.figure(figsize=(12,6))
    plt.subplot(2, 2, 1)
    plt.plot(epoch_numbers,norm_validation,color="red") 
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.title('Train losses')
    plt.subplot(2, 2, 2)
    plt.plot(epoch_numbers,norm_train,color="blue")
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.title('Validation losses')
    plt.subplot(2, 1, 2)
    plt.plot(epoch_numbers,norm_validation, 'r-',color="red")
    plt.plot(epoch_numbers,norm_train, 'r-',color="blue")
    plt.legend(['w=1','w=2'])
    plt.title('Train and Validation Losses')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    
    
    plt.show()

draw_graph(val_losses,train_losses,epochs)
            
 
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
        mg=cv2.resize(img,(224,224))
        mask_ind   = mask == 1
        cpy_img  = mg.copy()
        mg[mask==1 ,:] = (255, 0, 125)
        opac_image=(mg/2+cpy_img/2).astype(np.uint8)
        predict_name=batch_test[0]
        predict_path=predict_name.replace('images', 'predict')
        #predict_path2=predict_name.replace('AUG_IMAGE_DIR', 'predict')
        cv2.imwrite(predict_path,opac_image.astype(np.uint8))
        #cv2.imwrite(predict_path2,opac_image.astype(np.uint8))

predict(test_input_path_list)
