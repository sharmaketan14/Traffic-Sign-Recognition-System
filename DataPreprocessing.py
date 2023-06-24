#Contrast Normalization

#There will be a lot of contrast imbalance in our data of traffic signs so we will perform contrast normalization using CLAHE(Contrast Limited Adaptive Histogram Equilization) algorithm which will amplify the conttrast and make certain regions of image more visible to the model that we will build later. 

#To create a grayscale image.....

import cv2
from PIL import Image

class CLAHE:
    def __init__(self, clipLimit, tileGridSize=(4,4)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize
        
    def __call__(self, im):
        img = cv2.cvtColor(im, cv2.COLOR_RGB2YCrCb)[:,:,0]
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
        img = clahe.apply(img)



#Data Augmentation Pipeline
# Rotation -> Translation -> Shear Mapping -> Scaling


        

