import cv2
import numpy as np


def nothing(*arg):
    pass


cv2.namedWindow("result")  # создаем главное окно
cv2.namedWindow("settings")  # создаем окно настроек

cap = cv2.imread('11_masked_image.jpg')
blurred = cv2.medianBlur(cap, 15)
cap = blurred
# создаем 6 бегунков для настройки начального и конечного цвета фильтра
# createTrackbar ('Имя', 'Имя окна', 'начальное значение','максимальное значение','вызов функции при изменение бегунка'
cv2.createTrackbar('hue_1', 'settings', 0, 255, nothing)
cv2.createTrackbar('satur_1', 'settings', 0, 255, nothing)
cv2.createTrackbar('value_1', 'settings', 0, 255, nothing)
cv2.createTrackbar('hue_2', 'settings', 255, 255, nothing)
cv2.createTrackbar('satur_2', 'settings', 255, 255, nothing)
cv2.createTrackbar('value_2', 'settings', 255, 255, nothing)

while True:
    img = cap
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)# HSV формат изображения

    # считываем значения бегунков
    h1 = cv2.getTrackbarPos('hue_1', 'settings')
    s1 = cv2.getTrackbarPos('satur_1', 'settings')
    v1 = cv2.getTrackbarPos('value_1', 'settings')
    h2 = cv2.getTrackbarPos('hue_2', 'settings')
    s2 = cv2.getTrackbarPos('satur_2', 'settings')
    v2 = cv2.getTrackbarPos('value_2', 'settings')

    # формируем начальный и конечный цвет фильтра
    h_min = np.array((h1, s1, v1), np.uint8)
    h_max = np.array((h2, s2, v2), np.uint8)

    # накладываем фильтр на кадр в модели HSV
    thresh = cv2.inRange(hsv, h_min, h_max)

    cv2.imshow('result', thresh)

    ch = cv2.waitKey(5)
    if ch == 27:
        break

cap.release()
cv2.destroyAllWindows()
