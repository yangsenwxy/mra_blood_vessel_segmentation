#-----------------------------------
# Author:      Rudiger von Hackewitz 
# Date:        December 2018 
# 
# UNet Models that have been built for training and execution of the segmentation processes
# on the 9 different 2-dimensional image projections 
# U-NET model is based on work by Tobias Sterbak, https://www.depends-on-the-definition.com/unet-keras-segmenting-images 


import numpy as np
import math
from utils.support import create_black_image
import cv2
import os

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.optimizers import Adam

class MRA_UNets():
    def __init__(self, unet_path,d,h,w,no_projections, threshold):
        self._unet_path = unet_path 
        self._d = d
        self._h = h
        self._w = w
        self._no_projections = no_projections
        self._threshold = threshold 
        
        self._models = dict()  
    
    
    
    def load_model(self,p): 
        # use the adjusted image size of the projection for the neural network 
        h, w = self.get_image_projection_size (p)
        input_img = Input((h, w, 1), name='img')
        # now define the model 
        model = self.get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
        model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["binary_accuracy"]) 
        # If model had been trained before, load its parameters 
        model_par_file = self._unet_path+'model-Projection-'+p
        if os.path.exists(model_par_file): 
            model.load_weights(model_par_file)
        return model

    
    
    def get_models(self): 
        for i in range(self._no_projections):
            projection = str(i+1) 
            self._models[i] = self.load_model(projection)
      
    
    
    def segment_projections(self, mra_src_2Ds): 
        lst = list()
        for i in range(self._no_projections):
            projection = str(i+1) 
            h, w = self.get_image_projection_size (projection)
            ho, wo = self.get_image_size (projection)
            img = np.zeros((1, h, w, 1), dtype=np.float32) 
            if (ho, wo) != (mra_src_2Ds[i].shape[0], mra_src_2Ds[i].shape[1]):
                raise Exception ("Wrong input format for projection "+ projection)
            img [0,:ho,:wo,0] = mra_src_2Ds [i] / 255.0
            seg_map = self._models[i].predict(img, verbose=0)
            seg_map = seg_map.squeeze() # squeeze away empty dimensions (just two dimensions: height/width remain) 
            se = np.zeros((ho, wo), np.uint16) 
            se = (seg_map > self._threshold).astype(np.uint16)
            se = se[:ho,:wo] # trim it to the original image dimensions
            lst.append(se)
            
        return lst
      
    
    
    # return the original pair of (height, width) of the projected image: 
    def get_image_size (self, pr):
    
        if pr   == '1':
            return self._h, self._w
        elif pr == '2':
            return self._d, self._h
        elif pr == '3':
            return self._d, self._w
        elif pr == '4':
            return self._d, self._h+self._w-1
        elif pr == '5':
            return self._d, self._h+self._w-1
        elif pr == '6':
            return self._w, self._h+self._d-1
        elif pr == '7':
            return self._w, self._h+self._d-1
        elif pr == '8':
            return self._h, self._w+self._d-1
        elif pr == '9':
            return self._h, self._w+self._d-1
        else:
            raise Exception('Invalid projection number: '+ pr)
        

        
    # padding the image size before feeding it into the neural network 
    def get_image_projection_size (self, pr):
        height, width = self.get_image_size (pr)
        base = 64
        return int (math.ceil(height / base) * base), int (math.ceil(width / base) * base)
        
                
        
        
    def conv2d_block(self, input_tensor, n_filters, kernel_size=3, batchnorm=True):
        # first layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                   padding="same")(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
        # second layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",padding="same")(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    
    
    def get_unet(self, input_img, n_filters=16, dropout=0.5, batchnorm=True):
        # contracting path
        c1 = self.conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
        p1 = MaxPooling2D((2, 2)) (c1)
        p1 = Dropout(dropout*0.5)(p1)

        c2 = self.conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
        p2 = MaxPooling2D((2, 2)) (c2)
        p2 = Dropout(dropout)(p2)

        c3 = self.conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
        p3 = MaxPooling2D((2, 2)) (c3)
        p3 = Dropout(dropout)(p3)

        c4 = self.conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
        p4 = Dropout(dropout)(p4)
    
        c5 = self.conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
        # expansive path
        u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = self.conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

        u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = self.conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    
        u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = self.conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

        u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=3)
        u9 = Dropout(dropout)(u9)
        c9 = self.conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
        outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
        model = Model(inputs=[input_img], outputs=[outputs])
        return model
    
    
    
    
