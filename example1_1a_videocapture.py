import cv2
print(cv2.__version__)

cap = cv2.VideoCapture(0)

# CAP_SIZE = (1280,720)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_SIZE[0])
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_SIZE[1])

ret,im = cap.read() ## read ครั้งเดียว

print(im.shape)
print(type(im)) ## <class 'numpy.ndarray'>
print(im[0,0]) ## (b,g,r)
print(im[0,0,0]) ## b
print(type(im[0,0,0])) ## unit8 (เลขที่มันเก็บ 8 bits : 0 - 255 เท่านั้น ทศนิยมเก็บไม่ได้) ระวัง overflow คือ กรณีที่เลขเกิน 255 เลขจะเพี้ยน

cv2.imshow('camera',im) ## รับ unit8
cv2.imshow('blue channel',im[:,:,0])
cv2.imshow('green channel',im[:,:,1])
cv2.imshow('red channel',im[:,:,2])
cv2.waitKey()
cap.release()
