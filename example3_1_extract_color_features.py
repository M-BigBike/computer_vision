import numpy as np
import cv2
from matplotlib import pyplot as plt

#Download files from https://drive.google.com/file/d/1XdZLvORnCnfpyBYflh15I58VQrQdVlUe/view?usp=sharing

im = cv2.imread("C:/Users/Ez-Studio/computer_vision_660632034/dataset/SkinDetection/SkinTrain1.jpg")
mask = cv2.imread("C:/Users/Ez-Studio/computer_vision_660632034/dataset/SkinDetection/SkinTrain1_mask.jpg",0)

im_hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV) # เปลี่ยน รูป เป็น HSV
h = im_hsv[:,:,0] #ค่า h ทุกค่า
s = im_hsv[:,:,1] #ค่า ห ทุกค่า

h_skin = h[mask >= 128] # จะได้ค่า h ของ skin เท่านั้น
s_skin = s[mask >= 128] # จะได้ค่า s ของ skin เท่านั้น
h_nonskin = h[mask < 128] # จะได้ค่า h ของ non-skin เท่านั้น
s_nonskin = s[mask < 128] # จะได้ค่า s ของ non-skin เท่านั้น


cv2.imshow('image',im)
cv2.imshow('mask',mask)
cv2.imshow('hue',h)
cv2.imshow('saturation',s)

plt.plot(h_nonskin,s_nonskin,'b.') #show plot non-skin
plt.plot(h_skin,s_skin,'r.') #show plot skin
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
