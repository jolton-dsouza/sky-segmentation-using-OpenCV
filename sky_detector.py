import numpy as np

import cv2

import time

#import os

def SkyDetect(img_orig, row_scale=False, col_scale=False, visualize=False):
    try:
        contrast = 64
        alpha = 131*(contrast + 127)/(127*(131-contrast))
        gamma = 127*(1-alpha)
#        g_kernel = cv2.getGaborKernel((43, 43), 8.0, np.pi / 180, 10.0, 0.5, 0, ktype=cv2.CV_32F)

        if row_scale and col_scale:
            img_orig = cv2.resize(img_orig, (col_scale, row_scale))
            sky_mask = np.zeros([row_scale, col_scale])
        else:
            sky_mask = np.zeros([img_orig.shape[0], img_orig.shape[1]])  # Binary mask for sky detection(1's for sky region and 0's for non sky)

        start = time.time()

        contrasted_img = cv2.addWeighted(img_orig, alpha, img_orig, 0, gamma)

        # Convert to gray and find Otsu's threshold
        gray = cv2.cvtColor(contrasted_img, cv2.COLOR_BGR2GRAY)
        gray_gauss = cv2.GaussianBlur(gray,(3,3),0)

        thresh = cv2.threshold(gray_gauss,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thresh_img = cv2.threshold(gray_gauss, thresh[0], 255, cv2.THRESH_BINARY)[1]

        # Find contours in 30%of the image from top
        contours = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
        for i, contour in enumerate(contours):
            if contour[0][0][1] < np.round(0.30 * thresh_img.shape[0]):
                cv2.drawContours(sky_mask, contours, i, color=1, thickness=-1)

        stop = time.time()
        print("Time", stop - start)

        if visualize is True:
            img_orig[sky_mask == 0] = 0
            cv2.imshow('skyDetection', img_orig)
            cv2.waitKey(0)
        return sky_mask
    except Exception as e:
        print("Error in Sky Segmentaion: ", e)


img = cv2.imread('IMG_LOCATION')

sky_mask = SkyDetect(img,480,640, visualize= True)


#image_folder = 'IMG_LOCATION'
#video_name = 'SkyVideo1.mp4'
#
#images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
#frame = cv2.imread(os.path.join(image_folder, images[0]))
#height, width, layers = frame.shape
#
#video = cv2.VideoWriter(video_name, -1, 25, (width,height))
#
#for img in images:
#    s = SkyDetect(cv2.imread(os.path.join(image_folder, img)),  visualize= True)
#    video.write(s)
#    
#cv2.destroyAllWindows()
#video.release()

