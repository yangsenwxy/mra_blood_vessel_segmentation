# mra_blood_vessel_segmentation
New algorithm for the automatic segmentation of blood vessels in cerebral MRA scans.
The newly developed algorithm udertakes the following steps:
- Create a total of 9 2D image projections (3 front views and 6 edge views), to segment blood vessels in 2D space.
- CLAHE enhance the 9 image projections
- For each of the nine CLAHE projections, a neural network (U-NET) is trained to create blood vessel segments
- Reconstruct the 3D MRA raw segmentation, based on the 9 2D segmentations
- Apply image dilution to build up the full 3D segmentation map of blood vessels for the 3D source image

Process is configurable to work with various MRI scanner models and produced excellent results with performance metrics greater than 0.9 for precision, recall and F1 Score against manually segmented ground truth data.

Main program MRASegmentation.py can be started from the command line to initiate the process. Image slices for individual MRA scan need to be stored in source folder, results are stored in output folder.

License:
Copyright (C) 1993–2009 Louis Collins, McConnell Brain Imaging Centre, Montreal Neurological Institute, McGill University. Permission to use, copy, modify, and distribute this software and its documentation for any purpose and without fee is hereby granted, provided that the above copyright notice appear in all copies. The authors and McGill University make no representations about the suitability of this software for any purpose. It is provided “as is” without express or implied warranty. The authors are not responsible for any data loss, equipment damage, property loss, or injury to subjects or patients resulting from the use or misuse of this software package.

Original ICBM dataset:
Mazziotta, J., Toga, A., Evans, A., Fox, P., Lancaster, J., Zilles, K., … Mazoyer, B. (2001). A probabilistic atlas and reference system for the human brain: International Consortium for Brain Mapping (ICBM). Philosophical Transactions of the Royal Society of London. Series B, 356(1412), 1293–1322. http://doi.org/10.1098/rstb.2001.0915.

Semi-automatic segmentation dataset:
Song, B., Wen, P., Ahfock, T. & Li, Y. (2016) Numeric investigation of brain tumor influence on the current distributions during transcranial direct current stimulation. IEEE transactions on biomedical engineering, Vol.63, No.1, pp.176-187.
