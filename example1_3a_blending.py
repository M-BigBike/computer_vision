import numpy as np
import cv2

cap = cv2.VideoCapture(0)

ret,im = cap.read()
im_resized = cv2.resize(im, (640,360))
im_flipped = cv2.flip(im_resized, 1)
im0 = im1 = im2 = im3 = im_flipped #Ex. เอาภาพหลายภาพมาซ้อนๆกัน โดยใช้ภาพก่อนหน้า 5 ภาพ โดย im_flipped คือภาพปัจจุบัน

# while loop นำเอามาใช้เก็บภาพ ที่ย้อนหลัง update +1 step ไปเรื่อยๆ
while(True):
    im0 = im1
    im1 = im2
    im2 = im3
    im3 = im_flipped
    
    ret,im = cap.read()   ## อ่านภาพปัจจุบันใหม่
    im_resized = cv2.resize(im, (640,360))
    im_flipped = cv2.flip(im_resized, 1)

    #im_out = (0.2*im0 + 0.2*im1 + 0.2*im2 + 0.2*im3 + 0.2*im_flipped).astype(np.uint8) # Output เฉลี่ย 5 frame
    #im_out = ((im0+im1+im2+im3+im_flipped)/5).astype(np.uint8) # case นี้เป็น Overflow ที่ค่าที่เก็บเกิน 255
    im_out = (im_flipped*2)
    #im_out[im_out >= 255] = 255
    im_out = im_out.astype('uint8')
    cv2.imshow('camera',im_out) # Must be input with uint8 only

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
