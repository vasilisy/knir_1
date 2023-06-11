import cv2
import numpy as np


image = cv2.imread('14_masked_image.jpg')
blurred = cv2.medianBlur(image, 15)

#фильтры для каждого изображения, чтобы отображался квадрат
#11
#hsv_min = np.array((0, 50, 0), np.uint8)
#hsv_max = np.array((255, 255, 255), np.uint8)
#12
#hsv_min = np.array((0, 0, 0), np.uint8)
#hsv_max = np.array((255, 255, 130), np.uint8)
#13
#hsv_min = np.array((0, 0, 0), np.uint8)
#hsv_max = np.array((255, 255, 158), np.uint8)
#14
hsv_min = np.array((0, 0, 0), np.uint8)
hsv_max = np.array((255, 255, 212), np.uint8)


hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV ) # меняем цветовую модель с BGR на HSV
thresh = cv2.inRange(hsv, hsv_min, hsv_max ) # применяем цветовой фильтр
contours0, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow('thresh',thresh)
#cv2.drawContours(image, contours0, -1, (255, 0, 0), 2, cv2.LINE_AA)
#cv2.imshow('bl', image)

for cnt in contours0:
    rect = cv2.minAreaRect(cnt) # пытаемся вписать прямоугольник
    box = cv2.boxPoints(rect) # поиск четырех вершин прямоугольника
    box = np.intp(box) # округление координат

    if cv2.contourArea(cnt) > 3479 and cv2.contourArea(cnt) < 9000:
        if (box[0][0] and box[1][0] and box[2][0] and box[3][0] >= image.shape[0]//2) and (box[0][1] and box[1][1] and box[2][1] and box[3][1] <= image.shape[1]//2):
            cv2.drawContours(image,[box],0,(255,255,0),2) # рисуем прямоугольник
            print(cv2.contourArea(cnt))

cv2.imshow('contours', image) # вывод обработанного кадра в окно
#cv2.imwrite('14_square.jpg', image)


cv2.waitKey(0)
cv2.destroyAllWindows()
