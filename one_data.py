# -*- coding: utf-8 -*-
import numpy as np 
import tensorflow as tf 
import datetime
import matplotlib.pyplot as plt
import h5py
import json
from pprint import pprint

def h5_Open():
    f = h5py.File('D:/recipe/data.h5','r')   
    for key in f.keys():
        if key.find("train")>0:
            print(f[key].name)
            print(f[key].shape)
            #print(f[key].value)
    #print(f['stvecs_train'].value)
    image=f['ims_train'][1]
    vecs=f['stvecs_train'][1]
    print(image)
    print(vecs)
    one = h5py.File('one.h5','w')   #创建一个h5文件，文件指针是f  
    one['img'] = image                 #将数据写入文件的主键data下面  
    one['stvecs'] =vecs          #将数据写入文件的主键labels下面  
    one.close()  

    #x=np.array(f['impos_train'].value)
    #print(np.argwhere(x==2)[0][0])


def json_open(no):
    jsonFiles=["det_ingrs","layer1","layer2"]
    f=open("D:/recipe/"+jsonFiles[no]+".json","r")
    data=json.load(f)
    #pprint(data)
    print(len(data))    

def get_h5_data():
    return h5py.File('D:/recipe/data.h5','r')   
def get_recipe_id(file):
    x=np.array(file['ids_train'].value)
    x=list(map(str, x))
    x=list(map(lambda x:x[2:-1], x))
    return x
def get_image_name_index(file):
    #x=np.array(file['imnames_train'].value)
    y=np.array(file['impos_train'].value)
    return y

def get_vec_pos(file):
    x=np.array(file['rbps_train'].value)
    y=np.array(file['rlens_train'].value)
    return x,y
def load_image(file,pos,batch_size):
    images = file['ims_train'][pos:pos+batch_size-1]
    images=images.reshape((batch_size-1,256,256,3))
    return images
def load_vec(file,pos,len):
    vecs =file['stvecs_train'][pos-1:pos+len-2]
    return vecs
def merge():
    f=get_h5_data()
    #recipeIds=get_recipe_id(f)
    imageIndexs=get_image_name_index(f)
    vecPos,vecLen=get_vec_pos(f)
    countArray=np.count_nonzero(imageIndexs,axis=1)
    print(f['ims_train'][0].shape)
    x=f['ims_train'][0].reshape(256,256,3)
    print(x.shape)
    """
    for item in countArray:
        RecipeImage=load_image(f,0,item)
        print(RecipeImage.shape)
    
    for i, item in enumerate(vecLen):
        RecipeVec=load_vec(f,vecPos[i],item)
        print(RecipeVec.shape)
     """    

def get_training_batch(batch_no, batch_size,loaded_data):
    #initalize 
    cnt = 0
    realImages = np.zeros((batch_size, 256, 256, 3))
    captions =np.zeros((batch_size,20))
    #ims_train is image data but is too large  shape:(471557, 3, 256, 256)
    imagePos=np.array(loaded_data['impos_train'].value) #image position shape:(238459, 5)
    ingt2Vecs=np.array(loaded_data['ingrs_train'].value) # ingredint data shape:(238459, 20)
    #intPos=np.array(loaded_data['rbps_train'].value)# vec positon  shape:(238459,)
    #rLens=np.array(loaded_data['rlens_train'].value) #recipe vec count  shape:(238459,)
    for i in range(batch_no * batch_size, batch_no * batch_size + batch_size):
        imgArray=loaded_data['ims_train'][i]
        imgArray=imgArray.reshape(256,256,3)
        ones=np.ones((len(imgArray),len(imgArray[0]),3),dtype=np.float32)
        imgArray=np.add(imgArray,ones*(-128))  #減掉128   ecah pixel value=[-128,127]
        imgArray=np.multiply(imgArray,(1/(ones*128))) #除以 128   ecah pixel value=[-1,1]
        realImages[cnt,:,:,:] = imgArray

        index=np.argwhere(imagePos==(i+1))
        captions[cnt,:]=ingt2Vecs[index[0][0]-1]
        #captions.append(ingt2Vecs[index-1])
        print(len(captions))
    
        #print(captions.shape())
        cnt+=1
    

if __name__ == "__main__":
    #merge()
    h5_Open()
    #f=get_h5_data()
    #get_training_batch(3,128,f)
