import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from numba import jit
from GoruntuIsleme import Resim
import math

from pypylon import pylon

##################################################################
####################### FONKSİYONLAR #############################
##################################################################

@jit()
def agirlikMerkezi(listkp1, listkp2, ornekleme=5):
    #Örnekleme sayısı ile ilk x kadar noktaya göre açı çıkarılabilir.
    #Hassasiyet artar.

    midX0 = 0
    midX1 = 0
    midY0 = 0
    midY1 = 0

    noktaSayisi = len(list_kp1)

    if(noktaSayisi > ornekleme):
        noktaSayisi = ornekleme

    for i in range(0, noktaSayisi):
        midX0 += listkp1[i][0]
        midY0 += listkp1[i][1]
        midX1 += listkp2[i][0]
        midY1 += listkp2[i][1]

    # Noktaların Ortalama Değeri

    midX0 = midX0/noktaSayisi
    midY0 = midY0/noktaSayisi

    midX1 = midX1/noktaSayisi
    midY1 = midY1/noktaSayisi
    print("midX1 =", midX1)
    #---------------------------1
    #midX0, midY0

    print("X0= ", midX0, " - Y0= ", midY0, "\n",
          "X1= ", midX1, " - Y1= ", midY1)

    return midX0, midY0, midX1, midY1

@jit()
def vektorelMatch(list_kp1, list_kp2, ornekleme=5):
    # total uzunluk üzerinden benzeşme daha hızlı ve ihtiyaca göre sonuç verebilir.
    #vektör mü skaler mi?
    # açısal durumlarda hata oranı artıyor, vektörel olması daha önemli

    vect0 = 0
    vect1 = 0

    for i in range(ornekleme-1):
        deltaX0 = abs(list_kp1[i+1][0] - list_kp1[i][0])
        deltaY0 = abs(list_kp1[i+1][1] - list_kp1[i][1])

        deltaX1 = abs(list_kp2[i+1][0] - list_kp2[i][0])
        deltaY1 = abs(list_kp2[i+1][1] - list_kp2[i][1])

        vect0 += math.sqrt(math.pow(deltaX0, 2) + math.pow(deltaY0, 2))
        vect1 += math.sqrt(math.pow(deltaX1, 2) + math.pow(deltaY1, 2))

    if(vect1> vect0):
        match = vect0/vect1
    else:
        match = vect1/vect0

    return round(match*100, 2)

##################################################################
####################### FONKSİYONLAR #############################
##################################################################

#img2 = cv2.imread('opencv-feature-matching-image.jpg')

#silgi = cv2.imread("ornek.jpg")
cap = cv2.VideoCapture(1)

cap.set(3, 600)
cap.set(4, 800)

start = 0
while True:

    #------------CYCLE HESABI----------------
    cycleTime = (time.time() - start)*1000
    cycleTime = round(cycleTime, 2)
    print("CYCLE TIME= ", cycleTime)
    start = time.time()


    #---------- OBJE EŞLEŞTİRME -------------
    #Resim.LiveStream(1)
    img1 = cv2.imread("obje.jpg", 0)  # obje resim

    ret, img = cap.read()

    img2 = img          #img2 objenin aranıldığı alan

    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x:x.distance)

    p1 = matches[0]

    list_kp1 = []
    list_kp2 = []

    # Her bir match için
    for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Append to each list
        list_kp1.append((x1, y1))
        list_kp2.append((x2, y2))

    matchDots = 65  # iterasyona giren feature detects
    matchTest = vektorelMatch(list_kp1, list_kp2, ornekleme=5)
    print("MATCH TEST= ", matchTest)

    i = 5
    Resim.DrawCircle(img1,list_kp1[i][0], list_kp1[i][1], r=5)
    Resim.DrawCircle(img2, list_kp2[i][0], list_kp2[i][1], r=5)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:matchDots], None, flags=2)
    """cv2.imshow("Model Training", img1)
    cv2.waitKey(80)
    cv2.imshow("Model Training", img2)"""
    cv2.waitKey(1)




    ## Görselleştirme
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    img2 = Resim.DrawContour(img2, threshMin=90, threshMax=255)
    # img1 = Resim.DrawContour(img1)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:matchDots], None, flags=2)
    Resim.AddText(img3, y=400, text= (str(cycleTime))+"ms")
    Resim.AddText(img3, y=435, text= ("%"+ str(matchTest)))


    #print(type(matches[0]))
    #matches[0] formatı    <class 'cv2.DMatch'>

    """print("match 0 distance", matches[0].distance)
    print("match 0 imgIdx", matches[0].imgIdx)
    #print("match 0 mro", matches[0].mro)
    print("match 0 queryIdx", matches[0].queryIdx.pt)
    print("match 0 trainIdx", matches[0].trainIdx.pt)
    #cv2.DMatch.mr"""

    #time.sleep(0.1)
    #print(len(list_kp2))

    #print(list_kp1[0], ",", list_kp2[0])
    cv2.imshow("Model Training", img3)
    #cv2.imshow("Model Training", img2)
    cv2.waitKey(1)

    if (cv2.waitKey(10) & 0xFF == ord('e')):
        cap.release()
        cv2.destroyAllWindows()
        break
    #time.sleep(10)


"""plt.imshow(img3)
plt.show()"""
