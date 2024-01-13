import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import greycomatrix, greycoprops

#download images form https://drive.google.com/file/d/1JfJYr-qJvgt1Jyz-Gop-oRci-TipuQmb/view?usp=sharing

im = cv2.imread("C:/Users/Ez-Studio/computer_vision_660632034/dataset/TextureClassification//Beef//1.jpg")
im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
cv2.imshow("image",im)

#glcm = greycomatrix(im_gray, [5], [0], 256, symmetric=True, normed=True) #[5] คือ list ของระยะห่าง, [0] คือ list ของทิศทาง, 256 คือ ระดับของ intensity #Experiment 1
glcm = greycomatrix(im_gray, [5,10,15], [0,np.pi/2], 256, symmetric=True, normed=True) # หา 3 ระยะ 2 ทิศทาง #Experiment 2
print('GLCM Shape')
print(glcm.shape)

print('ASM')
print(greycoprops(glcm, 'ASM')) #ASM == Uniformity

print('contrast')
print(greycoprops(glcm, 'contrast'))

print('homogeneity')
print(greycoprops(glcm, 'homogeneity'))

print('correlation')
print(greycoprops(glcm, 'correlation'))

cv2.waitKey(0)
cv2.destroyAllWindows()

