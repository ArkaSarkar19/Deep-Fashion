import numpy as np
import pandas as pd 
import os
import cv2
from tqdm import tqdm


def create_label(number,vec_size):
    a=np.zeros((vec_size,1))
    a[number]=1
    return np.array(a)

def create_dataset_acs(IMG_SIZE=64,label_list=list(range(15))):
    #total images in training dataset=71094
    # mean image size 206,164
    DATASET_DIR='../fashion-data'
    DATASET_FILE_PATH='train.txt'

    train_file=open(os.path.join(DATASET_DIR,DATASET_FILE_PATH),'r')
    train_file_paths=train_file.read().split('\n')
    train_file.close()

    FILE_PATH='../fashion-data/labels.json'
    label_file=open(os.path.join(DATASET_DIR,FILE_PATH),'r')
    label_read=label_file.read()
    label_file.close()

    temp1=label_read.split('\n')
    label_file.close()

    labels_dict={}
    for i in temp1[1:-2]:
        key=i.split(':')
        if int(key[0]) in label_list:
            labels_dict[int(key[0])]=key[1][:-2]

    #IMG_SIZE=128
    training_data=[]
    k=0
    for img_path in tqdm(train_file_paths[:-1], position = 0 , desc = "Progress :"):
        k=k+1
        #img_path_full=os.path.join(DATASET_DIR+'/fashion-data/images',img_path+'.jpg')
        #img=cv2.imread(img_path_full,cv2.IMREAD_GRAYSCALE)
        #img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        
        #print(img)
        if int(img_path[0]) in label_list:
            img_path_full=os.path.join(DATASET_DIR+'/images',img_path+'.jpg')
            img=cv2.cvtColor(cv2.imread(img_path_full,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
            img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            label=create_label(int(img_path[0:2].split('/')[0]),len(label_list))
            training_data.append([np.array(img),label])
    
    shuffle(training_data)
    #np.save('acs_train_data_labels-{}.npy'.format(len(label_list)),training_data)
    return training_data,labels_dict