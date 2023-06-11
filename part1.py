import cv2
import numpy as np


image = cv2.imread('11.jpg')

# Шаг 1
#mask = np.zeros((815, 815), dtype = "uint8")
# Шаг 2
#cv2.circle(mask, (390, 390), 390, 255, -1)
#img = cv2.resize(image,(815,815),cv2.INTER_AREA)
#masked = cv2.bitwise_and(img,img,mask=mask)
# Шаг 3
#cv2.imshow("Masked image",bitwiseAnd)


imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('otsu method', thresh)


kernel = np.ones((3,3), np.uint8)
#opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
dilate_img = cv2.dilate(thresh, kernel, iterations=2)
#cv2.imshow("dilated", dilate_img)

blur_mask = cv2.medianBlur(dilate_img, 3)
#cv2.imshow('blur', blur_mask)

contours, hierarchy = cv2.findContours(blur_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
image_clone = imgray.copy()
cv2.drawContours(image_clone, contours, -1, (255, 255, 255), 2)
#cv2.imshow('contours', image_clone)


    #kernel_1 = np.ones((2,2), np.uint8)
    #kernel_2 = np.ones((5,5), np.uint8)
   # erode_img = cv2.erode(thresh_otsu_1, kernel_1, iterations=16)
   ## dilate_img = cv2.dilate(erode_img, kernel_2, iterations=3)
  #  cv2.imshow('erode_img', erode_img)
   # cv2.imshow('dilate_img', dilate_img)



#for cnt in contours0:
#        if len(cnt) > 5:
  #          ellipse = cv2.fitEllipse(cnt)
 #           cv2.ellipse(image,ellipse,(0,0,255),2)


#cv2.drawContours(image_clone, contours, -1, (0, 255, 0), 2, cv2.LINE_AA, hierarchy, 1)
#cv2.imshow('1', image_clone)
#cv2.drawContours(image_clone, contours, -1, (0, 255, 0), 2, cv2.LINE_AA, hierarchy, 2)
#cv2.imshow('2', image_clone)



#circles = cv2.HoughCircles(imgray,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=200,maxRadius=0)
#print(circles)
#circles = np.uint16(np.around(circles))
#for i in circles[0,:]:
    #draw the outer circle
 #   cv2.circle(imgray,(i[0],i[1]),i[2],(0,255,0),2)
    #draw the center of the circle
  #  cv2.circle(imgray,(i[0],i[1]),2,(0,0,255),3)
   # break


#cv2.imshow('done', imgray)

cv2.waitKey(0)
