import cv2
import numpy as np


import numpy as np

cap = cv2.VideoCapture(0)

while(True):
    ret,im = cap.read()

    ############ Resizing ############################
    TARGET_SIZE = (int(640*0.5),int(360*0.5)) # W,H
    im_resized = cv2.resize(im,TARGET_SIZE)
    print(im.shape)
    print(im_resized.shape) # W,H
    ##################################################

    ############ Flipping ############################
    im_flipped = cv2.flip(im_resized,1)
    ##################################################

    ############ Blurred ############################
    L = 25
    kernel = np.ones((L, L), np.float32) / L / L
    im_blurred = cv2.filter2D(im_flipped, -1, kernel)
    ##################################################
    
    ############ Median Filter #######################
    L = 25
    im_median = cv2.medianBlur(im_flipped, L)
    ##################################################

    cv2.imshow('original', im)
    cv2.imshow('modified', im_median) # Change variable name for displaying another images
    cv2.imshow('linear', im_blurred)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
