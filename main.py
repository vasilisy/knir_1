import cv2
import numpy as np


def circle_mask(image, center, radius):
    x = center[1]
    y = center[0]
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    masked = cv2.bitwise_and(image,image,mask=mask)
    return masked


def otsu_method(image):
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return thresh


#Поиск внутренних узоров, в данный момент не используется
def bradley_method(image):
    width = image.shape[0]
    height = image.shape[1]
    s = width//8
    s2 = s//2
    t = 0.15

    res = np.zeros((image.shape[0], image.shape[1]),np.uint8)
    imageIntegral = cv2.integral(image)
    imageIntegral = imageIntegral[1:,1:]

    #Находим границы для локальных областей
    for i in range(0, width):
        for j in range(0, height):
            x1 = i - s2
            x2 = i + s2
            y1 = j - s2
            y2 = j + s2

            if x1 < 0:
                x1 = 0
            if x2 >= width:
                x2 = width - 1
            if y1 < 0:
                y1 = 0
            if y2 >= height:
                y2 = height - 1

            count = (x2 - x1)*(y2 - y1)
            sum = imageIntegral[x2,y2] - imageIntegral[x1,y2] - imageIntegral[x2,y1] + imageIntegral[x1,y1]
            if (image[i,j] * count) < sum * (1-t):
                res[i,j] = 0
            else:
                res[i,j] = 255
    return res


def main():
    # Прочитываем изображения
    image = cv2.imread('11.jpg')

    hsv_min = np.array((68, 0, 0), np.uint8)
    hsv_max = np.array((255, 255, 255), np.uint8)
    hsv = cv2.cvtColor( image, cv2.COLOR_BGR2HSV ) # меняем цветовую модель с BGR на HSV
    thresh = cv2.inRange( hsv, hsv_min, hsv_max ) # применяем цветовой фильтр
    contours0, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imshow('thresh',thresh)
    #cv2.imwrite('14_thresh_hsv.jpg', thresh)
    areas = [cv2.contourArea(c) for c in contours0]
    sorted_areas = np.sort(areas)
    # Поиск основной окружности
    i = -1
    while True:
        cnt = contours0[areas.index(sorted_areas[i])] #the biggest contour
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        if radius < min(image.shape[0], image.shape[1])/2:
            center = (int(x),int(y))
            radius = int(radius)
            break
        else:
            i -= 1
    masked_image = circle_mask(image, center, radius)
    #cv2.imshow('contours', masked_image) # вывод обработанного кадра в окно
    #cv2.imwrite('14_masked_image.jpg', masked_image)

    # Делаем чб
    imgray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    # Сглаживаем
    blurred = cv2.medianBlur(imgray, 15)

    # Применяем OTSU - метод
    otsu = otsu_method(blurred)
    #cv2.imshow('otsu', otsu)
    #cv2.imwrite('14_otsu.jpg', otsu)

    contours, hierarchy = cv2.findContours(otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(masked_image,contours,-1, (0, 0, 0), 2, cv2.LINE_AA, hierarchy)
    cv2.imshow('result', masked_image)
    #cv2.imwrite('14_result.jpg', masked_image)

    # Применяем Брэдли - метод
    #bradley = bradley_method(blurred)
    #cv2.imshow('bradley', bradley)

    # Применяем обычный метод
    #r, threshold = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
    #cv2.imshow('threshold', threshold)


    cv2.waitKey(0)
    pass


if __name__ == '__main__':
    main()
    pass
