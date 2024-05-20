"""
-Basler kameradan alınan görüntüyü Robot pozisyonu olarak gönderir.


Yapılacaklar:
1)Kayıtlı resimden kontür seçimi +
2)Seçilen kontürün detaylandırılması +
3)Canlı görüntü üzerinden benzer kontür tespiti +
4)Benzerlik detaylandırmaları ve görselleştirilmesi +
5)Açı tespiti ve uygulaması +
6)Kütüphaneye eklenmesi
7)Step by step trig ile programın yönetilmesi
8)TCP/IP ile haberleşmesi ve robot programı
"""

# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from numba import jit
from GoruntuIsleme import Resim
import math
import socket

from pypylon import pylon
#--------------------FONKSİYONLAR-----------
def ContourAxisMatch(obj0,obj1,limit):
    f0 = 0
    f1 = 0
    f2 = 0
    f3 = 0
    fRes = 1000
    cnt = 0
    durum = False
    for i in range(4):
        f0 = abs(obj0[0] - obj1[i % 4]) / obj0[0] * 100
        f1 = abs(obj0[1] - obj1[(i + 1) % 4]) / obj0[1] * 100
        f2 = abs(obj0[2] - obj1[(i + 2) % 4]) / obj0[2] * 100
        f3 = abs(obj0[3] - obj1[(i + 3) % 4]) / obj0[3] * 100
        fResX = round((f0+f1+f2+f3)/4, 4)
        print(f0, f1, f2, f3, "fResX=", fResX)
        if(fResX<fRes):
            fRes = fResX
            cnt = i

    if(fRes < (100-(limit*100))):
        durum = True
    return durum, fRes, cnt

#--------------------MAIN-------------------
matchRate = 0.95
mmPx = 0.1122
pickAci = 270
pickL = 15 #mm cinsinden
#Camera ve robotun kalibrasyon yapıldığı noktalar değeri
cCX = 996.308 #Camera Calibration X
cCY = 576.539
rCX = 126.33 #Robot Calibration X
rCY = 73.22

# Create a socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Ensure that you can restart your server quickly when it terminates
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# Set the client socket's TCP "well-known port" number
well_known_port = 8881
sock.bind(('', well_known_port))

# Set the number of clients waiting for connection that can be queued
sock.listen(5)
hostname = socket.gethostname()
print(hostname)
ip_address = socket.gethostbyname(hostname)
print(ip_address, ":", well_known_port)
# loop waiting for connections (terminate with Ctrl-C)

# -----Öğretilen Parça Parametreleri---
obje = cv2.imread("robot1.jpg")# civata.jpg
objeKunye = np.zeros(10, dtype=float)

#-----Kontür Tespit Parametreleri------
objeGray = cv2.cvtColor(obje, cv2.COLOR_BGR2GRAY)
ret, objeThresh = cv2.threshold(objeGray, thresh=130, maxval=255, type=0)
objeContours, objeHierarchy = cv2.findContours(objeThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
objeAlan = 0
cntObj = 0
for i in range(0, len(objeContours)):
    cntObje = objeContours[i]# buradan contour seçimi yap
    x, y, w, h = cv2.boundingRect(cntObje)
    alanX = cv2.contourArea(cntObje)
    #---- Kontür Eliminasyon ve Tespit -----------
    if(x > 50):
        obje = cv2.drawContours(obje, [cntObje], contourIdx=0, color=(0, 255, 0),
                     thickness=2)  # contourIdx=-1 hepsi, contourIdx=0,1,2,3... spesifik contour
        if(alanX>objeAlan):
            #print(alanX)
            cntObj = i
            objeKunye[0] = alanX
            # Frame hariç bulunan en büyük contour
        #Resim.AddText(obje, text=str(i), x=x + 30, y=y + 30)
        #break
cntObje = objeContours[cntObj]
obje = cv2.drawContours(obje, [cntObje], contourIdx=0, color=(0, 255, 0), thickness=2)
objeKunye[1] = cv2.arcLength(cntObje, True)
#Resim.ShowImgPlot(obje)

#----Kontür Ağırlık Merkezi-------
massCenter = cv2.moments(cntObje)
cXO = int(massCenter['m10']/massCenter['m00'])
cYO = int(massCenter['m01'] / massCenter['m00'])
#obje = Resim.DrawCircle(obje, x=cXO, y=cYO, r=5)

#----Kontür Açısal Boundary------
rect = cv2.minAreaRect(cntObje)
box = cv2.boxPoints(rect)
box = np.int0(box)

#-----Merkez  Boundary Çizgileri------
d0, aci0 = Resim.VektorHesapla(p0x=cXO, p0y=cYO, p1x=box[0][0], p1y=box[0][1], detay=1)
d1, aci1 = Resim.VektorHesapla(p0x=cXO, p0y=cYO, p1x=box[1][0], p1y=box[1][1], detay=1)
d2, aci2 = Resim.VektorHesapla(p0x=cXO, p0y=cYO, p1x=box[2][0], p1y=box[2][1], detay=1)
d3, aci3 = Resim.VektorHesapla(p0x=cXO, p0y=cYO, p1x=box[3][0], p1y=box[3][1], detay=1)
#print(d0, d1, d2, d3)
objeKunye[2] = round(d0, 2)
objeKunye[3] = round(d1, 2)
objeKunye[4] = round(d2, 2)
objeKunye[5] = round(d3, 2)
objeKunye[6] = round(aci0, 2)
objeKunye[7] = round(aci1, 2)
objeKunye[8] = round(aci2, 2)
objeKunye[9] = round(aci3, 2)

print(objeKunye)
#cv2.imshow("test", obje)
#cv2.waitKey(0)



#--------------- CANLI KONTROL ------------------

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

camera.Open()

# Basler Settings
camera.ExposureAuto.SetValue('Off') #exposure ayarları
camera.ExposureTime.SetValue(30000)  #microsecond40k

camera.GainAuto.SetValue('Off') #gain ayarları
camera.Gain.SetValue(20)#16

camera.Width.Value = camera.Width.Max   #pixel frame ayarları
camera.Height.Value = camera.Height.Max
#camera.Get

#camera.Close()

print("DeviceClass: ", camera.GetDeviceInfo().GetDeviceClass())
print("DeviceFactory: ", camera.GetDeviceInfo().GetDeviceFactory())
print("ModelName: ", camera.GetDeviceInfo().GetModelName())
print("Pixels: ", camera.Width.Value, "x", camera.Height.Value)
Hardware_Trigger = False

if Hardware_Trigger:# reset registration
    camera.RegisterConfiguration(pylon.ConfigurationEventHandler(), pylon.RegistrationMode_ReplaceAll, pylon.Cleanup_Delete)

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
count = 1
start = 0

##----- CANLI KAMERA SABİTLERİ------------
objelerKunye = np.zeros((10, 10), dtype=float)
#print(objelerKunye)

newSocket, address = sock.accept()
print("Connected from", address)
# loop serving the new client

while camera.IsGrabbing():
    # ------------CYCLE HESABI----------------
    ms = (time.time() - start) * 1000
    start = time.time()
    fps = 1000 / ms
    print("FPS =", fps)
    #----------------TCP/IP HABERLEŞMESİ-----------------------

    receivedData = newSocket.recv(1024)
    print("Gelen Data=", receivedData)
    # print(type(receivedData))
    if not receivedData:
        newSocket.close()
        print("Disconnected from", address)
    # Echo back the same data you just received
    # newSocket.send(receivedData)

    # ----------------TCP/IP HABERLEŞMESİ-----------------------
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grabResult.GrabSucceeded():

        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()
        #img = cv2.rotate(img, 1)

        try:
            imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray, thresh=130, maxval=255, type=0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            okObjectNo = 0
            partNo = 0
            for i in range(0, len(contours)):
                cnt = contours[i]
                # ----Kontür Bondary Box----------
                x, y, w, h = cv2.boundingRect(cnt)
                if(x>50 and (w>30)):# ana frame kontürü ayıklama
                    alan = cv2.contourArea(cnt)
                    if(abs(alan-objeKunye[0])/objeKunye[0] < (1-matchRate) ):
                        print("alanlar benziyor-->", objeKunye[0], "-", alan)
                        # ----Kontür Çizgisi Hesabı-------
                        arcLength = cv2.arcLength(cnt, True)
                        if(abs(arcLength-objeKunye[1])/objeKunye[1] < (1-matchRate)):
                            print("yay uzunlugu benziyor-->", objeKunye[1], "-", arcLength)
                            partNo = partNo + 1

                            # ----Kontür Ağırlık Merkezi-------
                            massCenter = cv2.moments(cnt)
                            cX = int(massCenter['m10'] / massCenter['m00'])
                            cY = int(massCenter['m01'] / massCenter['m00'])
                            #img = Resim.DrawCircle(img, x=cX, y=cY, r=10)

                            # ----Kontür Açısal Boundary------
                            rect = cv2.minAreaRect(cnt)
                            box = cv2.boxPoints(rect)
                            box = np.int0(box)

                            obj0 = np.zeros(4, int)
                            obj1 = np.zeros(4, int)
                            obj1Aci = np.zeros(4, int)
                            obj0 = (objeKunye[2], objeKunye[3], objeKunye[4], objeKunye[5])

                            d0, aci0 = Resim.VektorHesapla(p0x=cX, p0y=cY, p1x=box[0][0], p1y=box[0][1], detay=1)
                            d1, aci1 = Resim.VektorHesapla(p0x=cX, p0y=cY, p1x=box[1][0], p1y=box[1][1], detay=1)
                            d2, aci2 = Resim.VektorHesapla(p0x=cX, p0y=cY, p1x=box[2][0], p1y=box[2][1], detay=1)
                            d3, aci3 = Resim.VektorHesapla(p0x=cX, p0y=cY, p1x=box[3][0], p1y=box[3][1], detay=1)

                            obj1 = (d0, d1, d2, d3)
                            obj1Aci = (aci0, aci1, aci2, aci3)
                            durum, rate, sira = ContourAxisMatch(obj0, obj1, limit=matchRate)
                            print(durum, rate, sira)

                            if(durum):

                                cv2.drawContours(img, [cnt], contourIdx=0, color=(0, 255, 0),
                                             thickness=2)
                                img = cv2.drawContours(img, [box], 0, (255, 255, 0), 2)
                                Resim.AddText(img, text=str(round(100-rate, 2)), x=box[(sira+2) % 4][0], y=box[(sira+2) % 4][1],)
                                img = Resim.DrawCircle(img, x=cX, y=cY, r=10)
                                aci = obj1Aci[sira] - objeKunye[6]

                                if(aci<0):
                                    aci = 360+aci

                                # ----Kontür Pick Point-------------
                                img, cX, cY = Resim.PointFromVector(img, cX, cY, aci=(pickAci+aci), boy=(pickL/mmPx))# piksel cinsinden
                                cXR = cX
                                cYR = img.shape[0] - cY

                                Resim.AddText(img, text=str(round(aci, 2)), x=box[(sira + 2) % 4][0],
                                              y=box[(sira + 2) % 4][1]+40)
                                img, xi, yi = Resim.PointFromVector(img, cX, cY, aci=aci-89, boy=160, type=2)
                                img, xi, yi = Resim.PointFromVector(img, cX, cY, aci=aci+1, boy=50, type=2)
                                Resim.AddText(img, text="("+str(partNo)+")", x=box[(sira + 2) % 4][0],
                                              y=box[(sira + 2) % 4][1]-40)
                                Resim.AddText(img, text=str(cXR) +"x"+ str(cYR), x=box[(sira + 2) % 4][0],
                                              y=box[(sira + 2) % 4][1] + 70)

                                rX, rY = Resim.Calibration(mmPx, cCX, cCY, rCX, rCY, cXR, cYR)

                                rX = round(rX, 2)
                                rY = round(rY, 2)
                                rZ = -20
                                aci = (aci-45) % 360
                                aci = round(aci)

                                print("Robot Pozisyonu=", cX, "x", cY)

                                #----------- ROBOT HABERLEŞMESİ -----------------
                                strRobot = "(" + str(rX) + "," + str(rY) + "," + str(rZ) + ",0,0," + str(aci) +",0)"
                                #strRobot = "(" + str(pickX) + "," + str(pickY) + "," + str(rZ) + ",0,0," + str(aci) + ",0)"
                                strRobot = str.encode(strRobot)
                                newSocket.send(strRobot)
                                break
                            else:
                                cv2.drawContours(img, [cnt], contourIdx=0, color=(0, 0, 255),
                                                 thickness=2)
                        else:
                            print("# yay uzunlugu benzemiyor!!!-->", objeKunye[1], "-", arcLength)
                            cv2.drawContours(img, [cnt], contourIdx=0, color=(0, 0, 255),
                                             thickness=2)
                    else:
                        print("# alanlar benzemiyor!!!-->", objeKunye[0], "-", alan)
                        cv2.drawContours(img, [cnt], contourIdx=0, color=(0, 0, 255),
                                         thickness=2)
        except:
            print("Bir hata oluştu...")

        img = Resim.Resize(img, 50)
        img = Resim.AddText(img, text="FPS="+str(round(fps, 2)), x=5, y=30,)
        cv2.imshow("Contour Based Detection - UniEyes", img)
        #Resim.ShowImgPlot(img)
        k = cv2.waitKey(1)
        if k == 27:
            # Resim.Kaydet(img, "obje.jpg")  # ESC tuşu son kareyi kaydeder.
            grabResult.Release()
            sock.close()
            # Resim.Kaydet(img,"test.png")
            # Resim.ShowImg(img)
            break