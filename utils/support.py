#-----------------------------------
# Author:      Rudiger von Hackewitz 
# Date:        December 2018 
#-----------------------------------

import configparser
import os
import numpy as np
import cv2

def read_ini_section (section): 
    config = configparser.ConfigParser()
    config.read('MRASegmentation.ini')
    return config[section] # read the section and return data 



def read_ini_parameter (section, par): 
    # return the parameters for further use 
    return section[par]


        
def create_black_image (h, w):
    # initialise black image with zeros and return result
    return np.zeros((h,w), np.uint16)


def convert_image_to_grey(image,h,w):
    # grab the image dimensions
    if image.shape[0] != h:
        logging.info('Wrong height') 
        raise ValueError
    if image.shape[1] != w:
        logging.info('Wrong width') 
        raise ValueError
    grey_image = create_black_image(h,w)
    grey_image[:h,:w] = image[:,:,0]
    # return the grey image (with a single colour channel)
    return grey_image



def get_filenames (pathname, ext):
    list_of_filenames = []
    for (dirpath, dirnames, filenames) in os.walk(pathname):
        for f in filenames:
            if f.endswith(ext): # just pick images with the correct extension
                list_of_filenames.append(f)

    # sort the files by the numbers in their filenames 
    list_of_filenames.sort()
    return list_of_filenames



def load_images (pathname,ext,d,h,w): 
    lf   = get_filenames (pathname,ext)
    limg = np.ndarray(shape=(d,h,w), dtype=np.uint16)
    i=0
    for f in lf: 
        img = cv2.imread(pathname+"/"+f)
        # convert image to grey 
        img = convert_image_to_grey(img,h,w)
        limg[i]=img
        i=i+1
    return limg 




def get_scores (seg_map, gt_map):
    diff = np.sign(gt_map.astype(int)) - (np.sign(seg_map.astype(int))*2)
    fn = (diff==1).sum()
    tp = (diff==-1).sum()
    fp = (diff==-2).sum()
    tn = (diff==0).sum()
    
    return fn,tp,fp,tn


def get_metrics (seg_map, gt_map):
    fn,tp,fp,tn = get_scores (seg_map, gt_map)
    p  = tp / (tp + fp)  # precision 
    r  = tp / (tp + fn)  # recall 
    f1 = 2*p*r/(p+r)     # F1 score

    #print('Precision:   ' + str(np.round(100*p,2))+'%')
    #print('Recall:      ' + str(np.round(100*r,2))+'%')
    #print('F1 Score:    ' + str(np.round(100*2*p*r/(p+r),2))+'%')
    #print('    ')
    
    return p, r, f1