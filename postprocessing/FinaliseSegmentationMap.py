#-----------------------------------
# Author:      Rudiger von Hackewitz 
# Date:        December 2018 
# 
# Finalisation of the process by adjusting some of the pixels against data in the 3D source file 


import numpy as np
from scipy.ndimage import generate_binary_structure, binary_dilation

class SegmentationMap():
    def __init__(self, SegMapRaw, mra_src_3D, min_hits, min_neighbours, pxl_threshold,dilution_threshold):
        self._3d_raw = SegMapRaw
        self._mra_src_3D = mra_src_3D
        self._min_hits = min_hits
        self._min_neighbours = min_neighbours
        self._pxl_threshold = pxl_threshold
        self._dilution_threshold = dilution_threshold
        
        
    def finalise (self): 
        m1 = self.cut_hits()
        m2 = self.consider_min_neighbours(m1)
        return m2
             
        
    def cut_hits(self): 
        return (self._3d_raw > self._min_hits - 1)
        
        
    def consider_min_neighbours (self, b_map): 
        n_map = (b_map * 0) # initialise empty map with zeros 
        
        idx = np.where(b_map == 1)
        
        offset = [-1,0,1]
        for x,y,z in zip(idx[0],idx[1],idx[2]): 
            for ox in offset:
                for oy in offset:
                    for oz in offset: 
                        try: 
                            n_map[x+ox,y+oy,z+oz] = n_map[x+ox,y+oy,z+oz] + 1 
                        except IndexError:
                            pass # Ignore when pixels on the boundaries of the cube raise index errors        
        
        added = n_map * (self._mra_src_3D.astype(int) > self._pxl_threshold).astype(int)
        k_neighbours = np.sign(b_map + (added >= self._min_neighbours).astype(np.uint16))
        
        # finally - apply dilution with set threshold for signal strength in the source image 
        # all neighbours + neigbhours of neighbours are candidates for dilution: 
        struct = generate_binary_structure(3, 2)
        fixed = binary_dilation(k_neighbours, structure=struct)
        
        
        #fixed = binary_dilation(k_neighbours)
        final = fixed * (self._mra_src_3D.astype(int) > self._dilution_threshold).astype(int)
     
        return final 
        