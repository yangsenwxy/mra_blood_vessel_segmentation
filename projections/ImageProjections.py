#-----------------------------------
# Author:      Rudiger von Hackewitz 
# Date:        December 2018 
# 
# Running the 2D image projections 


import numpy as np
from utils.support import create_black_image
import cv2

class MRAProjection():
    def __init__(self, ThreeDImg):
        self._3d_img = ThreeDImg
        
        self._d = ThreeDImg.shape[0]
        self._h = ThreeDImg.shape[1]
        self._w = ThreeDImg.shape[2]
        
        # build up image directory with image projections and CLAHE enhancements 
        self._img_projs = dict()
  
    
    # CLAHE-enhance the image projections 
    def do_clahe(self, img): 
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        output = clahe.apply(img.astype(np.uint8))
        return output.astype(np.uint16)
       
            
            
    def run_projections(self): 

        # 1st Projection 
        smax1  = np.amax(self._3d_img, axis=0)
        out1   = self.do_clahe(smax1)
        self._img_projs[1] = [smax1, out1]

        # 2nd Projection 
        smax2  = np.amax(self._3d_img, axis=1)
        out2   = self.do_clahe(smax2)
        self._img_projs[2] = [smax2, out2]

        # 3rd Projection 
        smax3  = np.amax(self._3d_img, axis=2)
        out3   = self.do_clahe(smax3)
        self._img_projs[3] = [smax3, out3]
        
        #################################
        # Now we run the edge projections 
        #################################
        
        black_box = np.zeros((self._d,self._h + self._w - 1,self._h + self._w - 1), np.uint16)
        for i in range(self._h):
            for j in range(self._w): 
                if j-i < self._w+1:
                    black_box[:,self._h+(j-i)-1,i+j] = self._3d_img[:,i,j]
                else: 
                    black_box[:,self._h+self._w-(j-i)-1,i+j] = self._3d_img[:,i,j]
        
        # Projection 4
        smax4  = np.amax(black_box, axis=1)
        out4   = self.do_clahe(smax4)
        self._img_projs[4] = [smax4, out4]
        
        # Projection 5
        smax5  = np.amax(black_box, axis=2)
        out5   = self.do_clahe(smax5)
        self._img_projs[5] = [smax5, out5]
        
        
        black_box = np.zeros((self._h,self._w+self._d-1,self._w+self._d-1), np.uint16)
        for i in range(self._d):
            for j in range(self._w): 
                if j-i < self._w+1:
                    black_box[:,self._d+(j-i)-1,i+j] = self._3d_img[i,:,j]
                else: 
                    black_box[:,self._d+self._w-(j-i)-1,i+j] = self._3d_img[i,:,j]
            
        # Projection 6
        smax6  = np.amax(black_box, axis=1)
        out6   = self.do_clahe(smax6)
        self._img_projs[6] = [smax6, out6]
        
        # Projection 7
        smax7  = np.amax(black_box, axis=2)
        out7   = self.do_clahe(smax7)
        self._img_projs[7] = [smax7, out7]
       
        
        black_box = np.zeros((self._w,self._d + self._h - 1,self._d + self._h - 1), np.uint16)
        for i in range(self._d):
            for j in range(self._h): 
                if j-i < self._h+1:
                    black_box[:,self._d+(j-i)-1,i+j] = self._3d_img[i,j,:]
                else: 
                    black_box[:,self._d+self._h-(j-i)-1,i+j] = self._3d_img[i,j,:]
     
        # Projection 8
        smax8  = np.amax(black_box, axis=1)
        out8   = self.do_clahe(smax8)
        self._img_projs[8] = [smax8, out8]
        
        # Projection 9
        smax9  = np.amax(black_box, axis=2)
        out9   = self.do_clahe(smax9)
        self._img_projs[9] = [smax9, out9]
       


    def reconstruct3DMRA(self, so_img_p): # 12 seconds 
        # initialise empty 3D 'black box' that will be segmented in the process below 
        black_box = np.zeros(shape=(self._d,self._h,self._w), dtype=np.uint16) 

        # Projection 1 
        p=0
        ix = np.equal(so_img_p[p], 1)
        idx = np.where(ix)
        # list of indices that flag blood vessel locations 
        indices = list(zip(idx[0],idx[1]))

        for idx in indices:
            y = idx[0]
            z = idx[1]
            max_val = np.max(self._3d_img[:,y,z])
            for i in np.where(np.greater_equal(self._3d_img[:,y,z], max_val))[0]:
                black_box[i,y,z] = black_box[i,y,z] + 1
        
        
        # Projection 2 
        p=1
        iy = np.equal(so_img_p[p], 1)
        idy = np.where(iy)
        # list of indices that flag blood vessel locations 
        indices = list(zip(idy[0],idy[1]))

        for idy in indices:
            x = idy[0]
            z = idy[1]
            max_val = np.max(self._3d_img[x,:,z])
            for i in np.where(np.greater_equal(self._3d_img[x,:,z], max_val))[0]:
                black_box[x,i,z]= black_box[x,i,z] + 1

        # Projection 3
        p=2
        iz = np.equal(so_img_p[p], 1)
        idz = np.where(iz)
        # list of indices that flag blood vessel locations 
        indices = list(zip(idz[0],idz[1]))

        for idz in indices:
            x = idz[0]
            y = idz[1]
            max_val = np.max(self._3d_img[x,y,:])
            for i in np.where(np.greater_equal(self._3d_img[x,y,:], max_val))[0]:
                black_box[x,y,i]= black_box[x,y,i] + 1

        # Projection 4      
        p=3
        xd = self._d
        yd = self._h + self._w - 1

        tilted_src = np.zeros((xd,yd,yd), np.uint16)
        tilted_blk = np.zeros((xd,yd,yd), np.uint16)

        for i in range(self._h):
            for j in range(self._w): 
                if j-i < self._w+1:
                    tilted_src[:,self._h+(j-i)-1,i+j] = self._3d_img[:,i,j]
                    tilted_blk[:,self._h+(j-i)-1,i+j] = black_box[:,i,j]
                else: 
                    tilted_src[:,self._h+self._w-(j-i)-1,i+j] = self._3d_img[:,i,j]
                    tilted_blk[:,self._h+self._w-(j-i)-1,i+j] = black_box[:,i,j]

        ix = np.equal(so_img_p[p], 1)
        idx = np.where(ix)
        for idx in list(zip(idx[0],idx[1])):
            x = idx[0]
            y = idx[1]
            max_val = np.max(tilted_src[x,:,y])
            for i in np.where(np.greater_equal(tilted_src[x,:,y], max_val))[0]:
                tilted_blk[x,i,y] = tilted_blk[x,i,y] + 1
        
        # Projection 5 
        p=4
        ix = np.equal(so_img_p[p], 1)
        idx = np.where(ix)
        for idx in list(zip(idx[0],idx[1])):
            x = idx[0]
            y = idx[1]
            max_val = np.max(tilted_src[x,y,:])
            for i in np.where(np.greater_equal(tilted_src[x,y,:], max_val))[0]:
                tilted_blk[x,y,i] = tilted_blk[x,y,i] + 1

        for i in range(self._h):
            for j in range(self._w): 
                if j-i < self._w+1:
                    black_box[:,i,j] = tilted_blk[:,self._h+(j-i)-1,i+j] 
                else: 
                    black_box[:,i,j] = tilted_blk[:,self._h+self._w-(j-i)-1,i+j] 
             
        # Projection 6      
        p=5
        xd = self._h
        yd = self._w + self._d - 1

        tilted_src = np.zeros((xd,yd,yd), np.uint16)
        tilted_blk = np.zeros((xd,yd,yd), np.uint16)

        for i in range(self._d):
            for j in range(self._w): 
                if j-i < self._w+1:
                    tilted_src[:,self._d+(j-i)-1,i+j] = self._3d_img[i,:,j]
                    tilted_blk[:,self._d+(j-i)-1,i+j] = black_box[i,:,j]
                else: 
                    tilted_src[:,self._d+self._w-(j-i)-1,i+j] = self._3d_img[i,:,j]
                    tilted_blk[:,self._d+self._w-(j-i)-1,i+j] = black_box[i,:,j]

        ix = np.equal(so_img_p[p], 1)
        idx = np.where(ix)
        for idx in list(zip(idx[0],idx[1])):
            x = idx[0]
            y = idx[1]
            max_val = np.max(tilted_src[x,:,y])
            for i in np.where(np.greater_equal(tilted_src[x,:,y], max_val))[0]:
                tilted_blk[x,i,y] = tilted_blk[x,i,y] + 1
        
        # Projection 7
        p=6
        ix = np.equal(so_img_p[p], 1)
        idx = np.where(ix)
        for idx in list(zip(idx[0],idx[1])):
            x = idx[0]
            y = idx[1]
            max_val = np.max(tilted_src[x,y,:])
            for i in np.where(np.greater_equal(tilted_src[x,y,:], max_val))[0]:
                tilted_blk[x,y,i] = tilted_blk[x,y,i] + 1

        for i in range(self._d):
            for j in range(self._w): 
                if j-i < self._w+1:
                    black_box[i,:,j] = tilted_blk[:,self._d+(j-i)-1,i+j] 
                else: 
                    black_box[i,:,j] = tilted_blk[:,self._d+self._w-(j-i)-1,i+j] 
             
     
        # Projection 8     
        p=7
        xd = self._w
        yd = self._h + self._d - 1

        tilted_src = np.zeros((xd,yd,yd), np.uint16)
        tilted_blk = np.zeros((xd,yd,yd), np.uint16)

        for i in range(self._d):
            for j in range(self._h): 
                if j-i < self._h+1:
                    tilted_src[:,self._d+(j-i)-1,i+j] = self._3d_img[i,j,:]
                    tilted_blk[:,self._d+(j-i)-1,i+j] = black_box[i,j,:]
                else: 
                    tilted_src[:,self._d+self._h-(j-i)-1,i+j] = self._3d_img[i,j,:]
                    tilted_blk[:,self._d+self._h-(j-i)-1,i+j] = black_box[i,j,:]

        ix = np.equal(so_img_p[p], 1)
        idx = np.where(ix)
        for idx in list(zip(idx[0],idx[1])):
            x = idx[0]
            y = idx[1]
            max_val = np.max(tilted_src[x,:,y])
            for i in np.where(np.greater_equal(tilted_src[x,:,y], max_val))[0]:
                tilted_blk[x,i,y] = tilted_blk[x,i,y] + 1
        
        # Projection 9
        p=8
        ix = np.equal(so_img_p[p], 1)
        idx = np.where(ix)
        for idx in list(zip(idx[0],idx[1])):
            x = idx[0]
            y = idx[1]
            max_val = np.max(tilted_src[x,y,:])
            for i in np.where(np.greater_equal(tilted_src[x,y,:], max_val))[0]:
                tilted_blk[x,y,i] = tilted_blk[x,y,i] + 1

        for i in range(self._d):
            for j in range(self._h): 
                if j-i < self._h+1:
                    black_box[i,j,:] = tilted_blk[:,self._d+(j-i)-1,i+j] 
                else: 
                    black_box[i,j,:] = tilted_blk[:,self._d+self._h-(j-i)-1,i+j] 
                                     
        return black_box        
        
 