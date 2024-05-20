"""
Yapılacaklar:
1)Kayıtlı resimden kontür seçimi
2)Seçilen kontürün detaylandırılması
3)Canlı görüntü üzerinden benzer kontür tespiti
4)Benzerlik detaylandırmaları ve görselleştirilmesi
5)Açı tespiti ve uygulaması
6)Step by step trig ile programın yönetilmesi
7)TCP/IP ile haberleşmesi ve robot programı
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from numba import jit
from GoruntuIsleme import Resim
import math

from pypylon import pylon

#---------------------FONKSİYONLAR-----------------------------------
def aciPCA(img):
    _, thresh = cv2.threshold(img, 250, 1, cv2.THRESH_BINARY_INV)
    mat = np.argwhere(thresh != 0)

    # let's swap here... (e. g. [[row, col], ...] to [[col, row], ...])
    mat[:, [0, 1]] = mat[:, [1, 0]]
    # or we could've swapped at the end, when drawing
    # (e. g. center[0], center[1] = center[1], center[0], same for endpoint1 and endpoint2),
    # probably better performance-wise

    mat = np.array(mat).astype(np.float32)  # have to convert type for PCA

    # mean (e. g. the geometrical center)
    # and eigenvectors (e. g. directions of principal components)
    m, e = cv2.PCACompute(mat, mean=np.array([]))
    print(m, e)
    # now to draw: let's scale our primary axis by 100,
    # and the secondary by 50

    center = tuple(m[0])
    endpoint1 = tuple(m[0] + e[0] * 100)
    endpoint2 = tuple(m[0] + e[1] * 50)

    red_color = (0, 0, 255)
    cv2.circle(img, center, 5, red_color)
    cv2.line(img, center, endpoint1, red_color)
    cv2.line(img, center, endpoint2, red_color)
    # cv2.imwrite("out.png", img)
    #cv2.imshow("asd", img)
    return img


#------------------------MAIN----------------------------------------






# -----Öğretilen Parça Parametreleri---
obje = cv2.imread("civata.jpg")
#-----Kontür Tespit Parametreleri------

objeGray = cv2.cvtColor(obje, cv2.COLOR_BGR2GRAY)
ret, objeThresh = cv2.threshold(objeGray, thresh=130, maxval=255, type=0)
objeContours, objeHierarchy = cv2.findContours(objeThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#print(len(cnt))
print(len(objeContours))
print(objeHierarchy[0])
for i in range(0, len(objeContours)):
    cntObje = objeContours[i]# buradan contour seçimi yap
    x, y, w, h = cv2.boundingRect(cntObje)
    #---- Kontür Eliminasyon-Tespit -----------
    if(x > 50):
        obje = cv2.drawContours(obje, [cntObje], contourIdx=0, color=(0, 255, 0),
                     thickness=2)  # contourIdx=-1 hepsi, contourIdx=0,1,2,3... spesifik contour
        Resim.AddText(obje, text=str(i), x=x + 30, y=y + 30)
        break
#----Alan Hesabı------------------
alanObje = cv2.contourArea(cntObje)

#----Kontür Çizgisi Hesabı-------
arcLengthObje = cv2.arcLength(cntObje, True)

#----Kontür Yaklaşık Hat Tespiti-
epsilonObje = 0.01 * cv2.arcLength(cntObje, True)
approxObje = cv2.approxPolyDP(cntObje, epsilonObje, True)
#obje = cv2.drawContours(obje, [approxObje], contourIdx=-1, color=(255, 0, 0), thickness=2)  # contourIdx=-1 hepsi, contourIdx=0,1,2,3... spesifik contour,

#----Kontür Genel Diagonal Hattı-
hullObje = cv2.convexHull(cntObje)
obje = cv2.drawContours(obje, [hullObje], contourIdx=-1, color=(0, 0, 255), thickness=2)  # contourIdx=-1 hepsi, contourIdx=0,1,2,3... spesifik contour
print("hullObjePoints= ", hullObje)

#----Kontür Açısal Boundary------
rect = cv2.minAreaRect(cntObje)
box = cv2.boxPoints(rect)
box = np.int0(box)
print("box Points=", box)
for i in range(4):
    obje = Resim.DrawCircle(obje, box[i][0], box[i][1], r=5)
    obje = Resim.AddText(obje,text=str(i), x=box[i][0], y=box[i][1])
obje = cv2.drawContours(obje, [box], 0, (0, 0, 255), 2)
aBoundXObje = rect[0][0]
aBoundYObje = rect[0][1]
aBoundWObje = rect[1][0]
aBoundHObje = rect[1][1]
aBoundAciObje = rect[2]

#----PCA ile Açı hesabı----------
aciPCA


#----Kontür Açısal Çizgisi-------
rows, cols = obje.shape[:2]
[vx, vy, xA, yA] = cv2.fitLine(cntObje, cv2.DIST_L2, 0, 0.01, 0.01)
lefty = int((-xA * vy / vx) + yA)
righty = int(((cols - xA) * vy / vx) + yA)
img = cv2.line(obje, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
obje = Resim.DrawCircle(obje, xA, yA, 3)
print("vx=", vx, "vy=", vy, "x=", xA, "y=", yA)
print("lefty=", lefty)
print("righty=", righty)

#----Eliptik Açı Hesabı---------
(xf, yf), (MA, ma), angle = cv2.fitEllipse(cntObje)
print("Eliptik Açı=", angle)

#----Kontür Bondary Box----------
xk, yk, wk, hk = cv2.boundingRect(cntObje)
#img = cv2.rectangle(img, (xk, yk), (xk + wk, yk + hk), (0, 170, 220), 2

#----Kontür Ağırlık Merkezi-------
massCenter = cv2.moments(hullObje)
cXO = int(massCenter['m10']/massCenter['m00'])
cYO = int(massCenter['m01'] / massCenter['m00'])
obje = Resim.DrawCircle(obje, x=cXO, y=cYO, r=5)




#------------DATA EXPORT---------------------
print("Kontür Alanı= ", alanObje, "px2",
      "Ağırlık Merkezi= ", cXO, "x", cYO,
      "Yay Uzunluğu=", round(arcLengthObje, 2), "px",
        #"Açısal Boundary- rect", rect,
        )
#obje = Resim.Resize(obje, 65)
#cv2.imshow("Multiple Contour Detection-Demo2", obje)
Resim.ShowImgPlot(obje)

k = cv2.waitKey(0)
if k == 27:
    # Resim.Kaydet(img, "obje.jpg")  # ESC tuşu son kareyi kaydeder.
    cv2.destroyAllWindows()
    # Resim.Kaydet(img,"test.png")
    # Resim.ShowImg(img)
    # print("OUTPUT=", text)



