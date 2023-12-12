#Download images from https://drive.google.com/file/d/1KqllafwQiJR-Ronos3N-AHNfnoBb8I7H/view?usp=sharing

#Bonus from this challenge https://docs.google.com/document/d/1q96VgmpJXlC95h9we-jiuxonEoYrZoqgTD2wjGk5TlI/edit?usp=sharing

import cv2

# การบ้าน: นับเหรียญแบบแยกสี ให้ได้ทกภาพ

def coinCounting('C:/Users/Ez-Studio/computer_vision_660632034/dataset/COIN/CoinCounting/coin1.jpg'):
    im = cv2.imread('C:/Users/Ez-Studio/computer_vision_660632034/dataset/COIN/CoinCounting/coin1.jpg')
    target_size = (int(im.shape[1]/2),int(im.shape[0]/2))
    im = cv2.resize(im,target_size)

    mask_yellow = cv2.inRange(im, (0, 100, 100), (100, 255, 255))
    mask_blue = cv2.inRange(im,(100,0,0),(255,100,100))

    mask_yellow = cv2.medianBlur(mask_yellow, 5)
    mask_blue = cv2.medianBlur(mask_blue, 5)

    contours_yellow, hierarchy_yellow = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_blue, hierarchy_blue = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    yellow = len(contours_yellow)
    blue = len(contours_blue)

    #print('Yellow = ',yellow)
    #print('Blue = ', blue)

    #cv2.imshow('Original Image',im)
    #cv2.imshow('Yellow Coin', mask_yellow)
    #cv2.imshow('Blue Coin', mask_blue)
    #cv2.waitKey()

    return [yellow,blue]


for i in range(1,11):
    print(i,":",coinCounting('.\CoinCounting\coin'+str(i)+'.jpg'))
