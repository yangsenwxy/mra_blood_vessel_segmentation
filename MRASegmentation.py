#-----------------------------------
# Author:      Rudiger von Hackewitz 
# Date:        December 2018 
# 
# Main process to run MRA image segmentation process
# Important: This process does NOT include training of
# the neural networks 


from utils.support import read_ini_section, read_ini_parameter, get_metrics
import projections.ImageProjections as ip 
import unets.UNetModels as un
import postprocessing.FinaliseSegmentationMap as pp 
import sys
import logging
import os 
import utils
import numpy as np
import cv2 


class MRASegmentation():
    def __init__(self):
        # Read initialisation parameters from ini-file 
        sc = read_ini_section('GLOBAL') 
        self._log_path    = read_ini_parameter(sc,'LogPath').strip()
        self._debug = (read_ini_parameter(sc,'Debug').upper() == 'YES')
        self._debug_path  = read_ini_parameter(sc,'DebugPath').strip()
        self._output_path  = read_ini_parameter(sc,'OutputPath').strip()
        
        
        # Source img sections, specifying characteristics of the source image slices 
        sc = read_ini_section('SOURCE') 
        self._source_path    = read_ini_parameter(sc,'SourcePath').strip()
        self._no_slices      = int(read_ini_parameter(sc,'NoSlices'))
        self._img_height     = int(read_ini_parameter(sc,'ImgHeight'))
        self._img_width      = int(read_ini_parameter(sc,'ImgWidth'))
        self._img_ext        = read_ini_parameter(sc,'ImgExt').strip()
        self._pxl_threshold  = int(read_ini_parameter(sc,'PxlThreshold'))
        
           
        # Neural Network (UNET) settings  
        sc = read_ini_section('UNET') 
        # path to the parameter model settings of the nine trained neural networks (U-Nets)
        self._model_path  = read_ini_parameter(sc,'ModelPath').strip()
        # Threshold used to define whether pixel is assigned blood vessel or not after UNET 
        # assigns each pixel a value between 0 and 1. By default it is set to 0.5 
        self._unet_threshold  = float(read_ini_parameter(sc,'Threshold'))
        
        
        # Parameters for post-processing 
        sc = read_ini_section('POST_PROCESSING') 
        self._min_hits           = int(read_ini_parameter(sc,'MinHits'))
        self._min_neighbours     = int(read_ini_parameter(sc,'MinNeighbours'))
        self._dilution_threshold = int(read_ini_parameter(sc,'DilutionThreshold'))

               
        # set up logging into a file for the process     
        # create empty log file if it does not exist, otherwise leave it (and append log entries to it
        logf = self._log_path
        open(logf,"a").close()
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d-%b-%Y %I:%M:%S %p: ',filename=logf,level=logging.DEBUG) 
        
         
        # OPTIONAL (only if GT available to calculate performance metrics)
        sc = read_ini_section('GROUND_TRUTH') 
        self._gt_path           = read_ini_parameter(sc,'GTPath').strip()
        self._gt_img_ext        = read_ini_parameter(sc,'ImgExt').strip()
        self._gt_pxl_threshold  = int(read_ini_parameter(sc,'PxlThreshold'))
        
        # The process has been designed with a total of 9 projections (9 orthogonal projections into the
        # MRA cube and 6 edge views into the MRA cube) 
        self.NO_PROJECTIONS = 9
        
        # pointer to the 9 instances of the UNet models (models not yet loaded) 
        self._unets = un.MRA_UNets(self._model_path,
                                   self._no_slices,
                                   self._img_height,
                                   self._img_width,
                                   self.NO_PROJECTIONS,
                                   self._unet_threshold)
    
    # if not done, set up subdirectories for each MRA record 
    def set_up_directories (self, path, lst): 
        for n in lst: 
            try:
                # Create target Directory
                os.mkdir(path + n)
            except FileExistsError:
                pass 
        
    
    def get_3d_data(self, path,ext): 
        r = utils.support.load_images (path,ext,self._no_slices,self._img_height,self._img_width)

        if r.shape != (self._no_slices,self._img_height,self._img_width): 
            logging.info('Input data are provided in the wrong format: ')
            raise ValueError
    
        return r
        
    # load the 3D source file images
    def get_3d_mra_source(self, mra_id): 
        r = self.get_3d_data(self._source_path + mra_id,self._img_ext)
            
        if self._debug: 
            r.tofile(self._debug_path + mra_id + '/Step1 Source-3D-Map')

        return r
        
    # Load the 3d ground truth images
    def get_3d_mra_gt (self, mra_id): 
        gt = self.get_3d_data(self._gt_path + mra_id,self._gt_img_ext)
            
        # all data above a certain threshold in the GT are considered 
        gt = (gt > self._gt_pxl_threshold).astype(int) 
        gt = (gt * 255)
        
        if self._debug: 
            gt.tofile(self._debug_path + mra_id + '/Step1 GT-3D-Map')
            
        return gt 
     
    # get the 3D output file images (predicted image maps)
    def get_3d_mra_output(self, mra_id): 
        out = self.get_3d_data(self._output_path + mra_id,'.jpg')      
        # all data above a certain threshold in the GT are considered 
        out = (out > self._gt_pxl_threshold).astype(int) 
        out = (out * 255)
        return out  
           
    def get_2d_mra_src_projs(self, mra_src_3D):     
        projs = ip.MRAProjection(mra_src_3D)
        projs.run_projections()
        p = projs._img_projs
        
        # Write the files into the debug directory (only if debug mode switched on) 
        if self._debug: 
            for i in range(self.NO_PROJECTIONS):   # work through the nine different 2D image perspectives
                cv2.imwrite(self._debug_path+self._mra_id+'/Step2 2D-Source-Projection-'+str(i+1)+'.jpg',p[i+1][0])
                cv2.imwrite(self._debug_path+self._mra_id+'/Step2 2D-Source-Projection-'+str(i+1)+'-CLAHE.jpg',p[i+1][1])
         
        # just return the list of CLAHE enhanced image projections, as they will be fed into the neural network 
        return [p[i+1][1] for i in range(self.NO_PROJECTIONS)]
    
    # for the training of the neural networks we need to create the image projections of the 3D ground truth data
    def get_2d_mra_gt_projs(self, mra_gt_3D):     
        projs = ip.MRAProjection(mra_gt_3D)
        projs.run_projections()
        p = projs._img_projs
        
        # Write the files into the debug directory (only if debug mode switched on) 
        if self._debug: 
            for i in range(self.NO_PROJECTIONS):   # work through the nine different 2D image perspectives
                cv2.imwrite(self._debug_path+self._mra_id+'/Step2 2D-GT-Projection-'+str(i+1)+'.jpg',255*p[i+1][0])
         
        # just return the raw image projections, CLAHE is not required for the ground truth data 
        return [p[i+1][0] for i in range(self.NO_PROJECTIONS)]
                
        
    def get_2d_mra_seg_projs(self, mra_src_2Ds):
        r = self._unets.segment_projections(mra_src_2Ds)
        
        # Write the files into the debug directory (only if debug mode switched on) 
        if self._debug: 
            for i in range(self.NO_PROJECTIONS):   # work through the nine different 2D image perspectives
                cv2.imwrite(self._debug_path+self._mra_id+'/Step3 2D-Segmentation-'+str(i+1)+'.jpg', r[i]*255)
         
        return r 
    
        
    def reconstruct_3d_segmentation(self, mra_seg_2Ds,mra_src_3D): 
        projs = ip.MRAProjection(mra_src_3D)
        r = projs.reconstruct3DMRA(mra_seg_2Ds)  
        
        # cross check that the segmentation map is in the right shape 
        if r.shape != (self._no_slices,self._img_height,self._img_width): 
            logging.info('3D raw segmentation map is in the wrong format: '+ str(shp))
            raise ValueError
            
        # include a cross validation / count check to verify sanity of the pixel values in segmentation map: 
        hits = [ np.sum(r == i) for i in range(self.NO_PROJECTIONS + 1) ]
        s1 = sum([x for x in hits])
        s2 = self._no_slices * self._img_height * self._img_width
        if s1 != s2: 
            logging.info('Pixel Count in Segmentation Map does not match with expected pixel count in Source')
            raise ValueError
        
        if self._debug: 
            r.tofile(self._debug_path + self._mra_id + '/Step4 Raw-3D-Segmentation-Map')
        
        return r
        
        
    def post_processing(self, mra_seg_3D_raw,mra_src_3D): 
        postp = pp.SegmentationMap(mra_seg_3D_raw, mra_src_3D, self._min_hits, self._min_neighbours, 
                                   self._pxl_threshold,self._dilution_threshold) 
        r = postp.finalise()
        
        if self._debug: 
            r.tofile(self._debug_path + self._mra_id + '/Step5 Final-3D-Segmentation-Map')
        
        return r

    
    def write_output(self,mra_seg_3D): 
        for i in range(self._no_slices): 
            out = 255*mra_seg_3D[i]
            cv2.imwrite(self._output_path+self._mra_id+'/'+self._mra_id+str(i).zfill(4)+'.jpg',out)
 

    def performance_metrics(self,mra_seg_3D):
        if os.path.exists(self._gt_path + self._mra_id): 
            # Load the ground truth data
            gt = self.get_3d_mra_gt (self._mra_id)
            
            # Calculate the performance metrics of the segmentation 
            p, r, f1 = get_metrics(mra_seg_3D, gt)     
            
            logging.info('Precision:   ' + str(np.round(100*p,2))+'%')
            logging.info('Recall:      ' + str(np.round(100*r,2))+'%')
            logging.info('F1 Score:    ' + str(np.round(100*f1,2))+'%')
            print('Precision:   ' + str(np.round(100*p,2))+'%')
            print('Recall:      ' + str(np.round(100*r,2))+'%')
            print('F1 Score:    ' + str(np.round(100*f1,2))+'%')
        
 
    def get_segmentation_map(self, mra_id):
        
        logging.info('***************************************************************')
        logging.info('Starting the blood vessel segmentation process for ID ' + mra_id)
        logging.info('***************************************************************')
         
        self._mra_id = mra_id
        
        # Read in 3D MRA source data 
        mra_src_3D = self.get_3d_mra_source(mra_id) 
        logging.info('Step 1 - 3D source data loaded')
        
        # construct the nine 2D MRA projections 
        mra_src_2Ds = self.get_2d_mra_src_projs(mra_src_3D) 
        logging.info('Step 2 - 2D image projections created')
        
        # run the nine unet processes across the image source projections 
        mra_seg_2Ds = self.get_2d_mra_seg_projs(mra_src_2Ds) 
        logging.info('Step 3 - 9 image segmentation processes completed')
        
        # Reconstruct the 3D segmented raw image 
        mra_seg_3D_raw = self.reconstruct_3d_segmentation(mra_seg_2Ds,mra_src_3D) 
        logging.info('Step 4 - 3D segmentation map for blood vessels reconstructed')
        
        # Postprocessing 
        mra_seg_3D = self.post_processing(mra_seg_3D_raw,mra_src_3D) 
        logging.info('Step 5 - Postprocessing for 3D segmentation map complete')
        
        # Write segmentation map into output directory 
        self.write_output(mra_seg_3D) 
        logging.info('Step 6 - Writing 3D segmentation data into output directory')
        
        # Calculate performance metrics (if ground truth data is available) 
        if os.path.exists(self._gt_path + self._mra_id): 
            self.performance_metrics(mra_seg_3D)
        
        return mra_seg_3D
   
    # get all the source IDs (based on the directory names in the source folder) 
    def get_mra_source_ids(self): 
        # run through all the records in the source directory 
        return [f.name for f in os.scandir(self._source_path) if f.is_dir() and f.name[0] != '.'] 
        
# run the MRA segmentation process 
def run_it ():
    app = MRASegmentation()
    
    # get all the source ids in the source folder 
    lst = app.get_mra_source_ids ()
    
    # process all the ids that are provided via the command line and are available in source directory: 
    if len(sys.argv) > 1: 
        lst = list(set(lst) & set(sys.argv[1:])) 
    
    if len(lst) == 0:
        print ('No MRA scans are available and nothing can be processed. Set up source data in directory '+app._source_path)
    
    else:
        logging.info('Initialising the process...')
        # As required, set up debug subdirectories for each record id
        app.set_up_directories (app._output_path, lst)
        if app._debug:
            app.set_up_directories (app._debug_path, lst)
        
        # Load the parameters of the nine UNet models 
        app._unets.get_models()
        logging.info('Initialisation complete and UNET models loaded')
        
        # Now segment the source data for each provided record id: 
        for nme in lst: 
            try:
                print('Processing Segmentation Map for Record '+nme+' ...') 
                r = app.get_segmentation_map(nme)
                print('Record '+nme+' segmented successfully.') 
            except Exception as e:
                logging.fatal(e, exc_info=True) 
                print('Record '+nme+' could not be segmented. Check the log/debug files for errors.') 
    
        logging.info('Overall process complete.')
run_it() 
