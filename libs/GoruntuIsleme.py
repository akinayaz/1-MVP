#Genel görsel işlemler metotu, Resim sınıfı ana sınıftır.
#Kütüphaneyle genellikle LOGITECH C270 Kamera kullanılmıştır.
#C270 Kamera 720p 30 fps özellikli  33ms de 1
#Camera 1280x720 maksimum çözünürlük

#Kullanılan BASLER Kamera acA1920-40um 41 fps monocolor 1920x1200px

import time, math,sqlite3, numpy as np, cv2, datetime, matplotlib.pyplot as plt
import pandas as pd
from PIL import ImageFont, ImageDraw, Image

#from pyzbar import pyzbar
import imutils

#from pypylon import pylon

from numba import jit_module, jitclass, jit
#import pytesseract


#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#Yenilik fikirleri:
#   -Öğretilen nesnelerin database kaydı
#   -Harici resimle öğrenme
#   -Modül altına kamera construction'ı  eklenmeli.

class Resim(object):
    mm1px = 0.0894      #--   endeskop kamera fix fov
    def __init__(self, _resim):
        self.resim =_resim
        self.kunye = np.zeros((10, 4), float)  #[0][0]= ort renk blue, [0][3]=ort renk mono
        self.nesneTanila()
    #----Class Variables----?
    #
    #-----------------------
    def imgSize(self):
        return np.size(self.resim, 1), np.size(self.resim, 0)

    def nesneTanila(self):# Künye oluşturma işlemlerinin tümü burada işlenmeli!
        #print("{} tanimlaniyor".format(nesne[2]))
        self.kunye[0] = self.OrtRenkTonu(self.resim)   #bgrm   // blue, green, red, monocolor
        self.kunye[1] = self.RenkSkalaMin(self.resim)  #bgrm   // blue, green, red, monocolor
        self.kunye[2] = self.RenkSkalaMax(self.resim)  #bgrm   // blue, green, red, monocolor
        self.kunye[3] = self.KenarOlculeri(self.resim, tip=0)# kenar uzunluğu
        self.kunye[4] = self.KenarOlculeri(self.resim, tip=1)# kenarların orjinle yaptığı açı.
        self.kunye[5] = self.objeGeometrik(self.resim, self.kunye, Resim.mm1px)# alan,çevre,konumX, konumY
        self.kunye[6] = self.Obje4Kose(self.resim, x=True)# parçanın x köşeleri
        self.kunye[7] = self.Obje4Kose(self.resim, x=False)
        return self.kunye

#Classmethodları--------------------------------------------------------------------------------------------------------

    @classmethod ###
    def ShowImg(cls, resim, text="aciklama yok"): # ÇÖZÜNÜRLÜK EKLE AÇIKLAMAYA,RGB Mİ ONU EKLE
        while True:
            cv2.imshow(text, resim)
            if (cv2.waitKey(10) & 0xFF == ord('e')):
                cv2.destroyAllWindows()
                break

    @classmethod
    def Threshold(cls, img, min=127, max=255, type=0):
        if(type==0):
            retval, imgT = cv2.threshold(img, min, max, cv2.THRESH_BINARY) # siyah beyaz, ölçüm için iyi
        elif(type==1):
            retval, imgT = cv2.threshold(img, min, max, cv2.THRESH_TOZERO) #siyah beyaz gri
        elif(type==2):
            retval, imgT = cv2.threshold(img, min, max, cv2.THRESH_TRIANGLE) #hatalı
        elif(type==3):
            retval, imgT = cv2.threshold(img, min, max, cv2.THRESH_TRUNC) #grayscale, netlik ve görüntü inceleme için iyi
        elif(type==4):
            retval, imgT = cv2.threshold(img, min, max, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)

        return imgT

    @classmethod # bu metodun siyah beyazı da yapılabilir ilk olarak.------------------------------------------------------------buradasın
    def BlobYakalaRGB(cls, img, R, G, B, yuzde, min, max):
        #yuzde değeri renge benzerlik oranını ayarlar.

        #min = minimum eşleşen piksel sayısını ifade eder.
        #max = maksimum eşleşen piksel sayısını ifade eder.


        # Fotoğraf üzerinde ölçekleme yap;
        return

    @classmethod
    def BlobYakalaMono(cls, img, min, max):
        # Ton 0-255 arası olacak.
        # Yüzde miktarı ile aralık belirlenecek.
        # Min ve max değerleri arasındaki değerler birleştirilecek.
        x = img.shape[1]
        y = img.shape[0]

        dots = []
        blobs = []

        x = img.shape[0]
        y = img.shape[1]
        return

    @classmethod
    def SmartSensorDetection(cls):
        result = np.zeros((4), int)
        for i in range(10):
            try:
                frame = cv2.VideoCapture(i)
                ret, img = frame.read()
                source = i
                shape = Resim.Size(img)
                result[0] = source
                if (shape[2] > 1):
                    result[1] = 1
                else:
                    result[1] = 0
                result[2] = shape[0]
                result[3] = shape[1]
                return  result # Source, color, pixelX, pixelY      /color=0 mono, color=1 RGB
            except:
                print(i)

    @classmethod ###
    def objeGeometrik(cls, resim, kunye, mmpx):
        geos = np.zeros((4), float)
        x1 = resim[1]#xMin
        y1 = resim[2]#xMiny
        x2 = resim[3]#xMax
        y2 = resim[4]#xMaxy
        x3 = resim[5]#yMinx
        y3 = resim[6]#yMin
        x4 = resim[7]#yMaxx
        y4 = resim[8]#yMax

        #Optimize-1 alan formülü=
        xRoi = (x2-x1)*mmpx
        yRoi = (y4-y3)*mmpx
        roiAlan = xRoi * yRoi
        print("ROI", xRoi, yRoi)

        #if(kunye[4][0] < 90):
            #print("YON= SAG")
        #else:
            #print("YON= SOL")

        koseAlanlari = 0
        #Optimize-2 alan formülü:
        kose1 = ((x3-x1)*mmpx) * ((y1-y3)*mmpx) / 2
        kose2 = ((x2-x3)*mmpx) * ((y2-y3)*mmpx) / 2
        kose3 = ((y4-y2)*mmpx) * ((x2-x4)*mmpx) / 2
        kose4 = ((x4-x1)*mmpx) * ((y4-y1)*mmpx) / 2
        koseAlanlari = kose1 + kose2 + kose3 + kose4
        alan = roiAlan - koseAlanlari
        #print(kose1, kose2, kose3, kose4, koseAlanlari)
        print("ROIAlan= " + str(roiAlan))
        #print("KoseAlanlari=" + str(koseAlanlari))
        print("ALAN= " + str(alan))

        geos[0] = alan
        geos[1] = (kunye[3][0] + kunye[3][1] + kunye[3][2] + kunye[3][3]) # çevre hesabı
        geos[2] = (x1 + x2)/2
        geos[3] = (y3 + y4)/2
        return geos

    @classmethod ###
    def Size(cls, resim):#Title için string değer gönderir.
        #size = np.zeros((2), int)
        x = np.size(resim, 1)
        y = np.size(resim, 0)
        c = np.size(resim, 2)
        #x=str(x)+"x"+str(y)
        #print(x,y)
        return (x,y,c)

    @classmethod
    def GetAngle(cls, p1, mid, p2):
        ang = math.degrees(math.atan2(p2[1] - mid[1], mid[0] - mid[0]) - math.atan2(p1[1] - mid[1], p1[0] - mid[0]))
        return ang + 360 if ang < 0 else ang

    @classmethod
    def DrawCircle(cls, img, x, y, r, B=0, R=255, G=0, thickness = 2):
        img = cv2.circle(img, (int(x),int(y)), r, (B,G,R), thickness )
        return img

    @classmethod
    def Obje4Kose(cls, resim, x=True):
        xMin = resim[1]
        xMiny = resim[2]
        xMax = resim[3]
        xMaxy = resim[4]
        yMinx = resim[5]
        yMin = resim[6]
        yMaxx = resim[7]
        yMax = resim[8]

        if(x):
            return xMin, xMax, yMinx, yMaxx
        elif not (x):
            return xMiny, xMaxy, yMin, yMax

    @classmethod
    def DrawGeos(cls, resim, foto, kunye):
        xMin = resim[1]
        xMiny = resim[2]
        xMax = resim[3]
        xMaxy = resim[4]
        yMinx = resim[5]
        yMin = resim[6]
        yMaxx = resim[7]
        yMax = resim[8]

        resim2 = foto
        midX = kunye[5][2]
        midY = kunye[5][3]
        aciX = 0
        aciY = aciX + 90

        cv2.line(resim2, (yMinx, yMin), (xMax, xMaxy), (255), 2)
        cv2.line(resim2, (xMax, xMaxy), (yMaxx, yMax), (255), 2)
        cv2.line(resim2, (yMaxx, yMax), (xMin, xMiny), (255), 2)
        cv2.line(resim2, (xMin, xMiny), (yMinx, yMin), (255), 2)
        #cv2.circle(resim2, (midX, midY), thickness=3, radius=10, color=(0,255,0))
        cv2.circle(resim2, (int(midX), int(midY)), 3, (200, 255, 200), -1)
        x2 = cls.GetPointFromVector(50, aciX)
        y2 = cls.GetPointFromVector(50, aciY)


        Resim.ShowImg(resim2)
        return resim2

    @classmethod
    def GenerateRotated(cls, img, aci=15):
        images = []
        for angle in np.arange(0, 360, aci):
            rotated = imutils.rotate_bound(img, angle)
            images.append(rotated)
            #cv2.imshow("Rotated (Correct)", rotated)
            #cv2.waitKey(1)
        return images

    @classmethod
    def GetPointFromVector(cls, l, a):  #return dX, dY ~= deltaX,Y
        x1 = l * math.cos(a)
        y1 = l * math.sin(a)
        cord2 = np.zeros((2), int)
        cord2[0] = x1
        cord2[1] = y1
        return cord2

    @classmethod
    def PointFromVector(cls, img, pX, pY, aci, boy=50, type=1):
        aciRad = math.radians(aci+90)
        sin = math.cos(aciRad)
        cos = math.sin(aciRad)
        resX = round(pX + (cos*boy))
        resY = round(pY + (sin*boy))
        if(type==0):
            return resX, resY
        if(type==1):
            img = Resim.DrawCircle(img, resX, resY, r=5)
            return img, resX, resY
        if(type==2):
            img = Resim.DrawLine(img, pX, pY, resX, resY)
            return img, resX, resY


    @classmethod    
    def DePolarization(cls, ):
        return 0

    @classmethod
    def Render(cls, img, scale=200, blur=15, thresholdMin=127, thresholdMax=255,thresholdType=0, blurType=0, resizeMethod=0):
        img = Resim.Resize(img, yuzde=scale, method=resizeMethod)# resize metotu üzerinde de oynanabilir,0-3-4
        img = Resim.Blur(img, k=blur, type=blurType)
        img = Resim.Threshold(img, type=thresholdType, min=160, max=255)#127-255
        return img

    @classmethod
    def Blur(cls, img, k=3, type=0):
        if(type==0):
            img = cv2.blur(img, (k, k))
        elif(type==1):
            img = cv2.GaussianBlur(img, (k, k), 0)
        elif(type==2):
            img = cv2.medianBlur(img, k)
        elif(type==3):
            img = cv2.bilateralFilter(img, k, 75, 75)
        return img

    @classmethod
    def BlobAnaliz(cls, resim, mmpx=1, sizeMin=3000, sizeMax=10000, minThreshold=10, maxThreshold = 200):
        img = resim
        im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # im = cv2.imread("blob3.jpg", cv2.IMREAD_GRAYSCALE)

        params = cv2.SimpleBlobDetector_Params()
        params.minDistBetweenBlobs = 5

        params.minThreshold = minThreshold  # 10
        params.maxThreshold = maxThreshold  # 200

        params.filterByArea = True
        params.minArea = sizeMin  # 300 - R=20px-----denemeOK1-11000---Fairy denemesi 12000
        params.maxArea = sizeMax  # 500 -  R=25px----denemeOK1-15000---Fairy denemesi 15000

        params.filterByCircularity = False
        params.minCircularity = 0.01
        # params.maxCircularity = 0.8

        params.filterByConvexity = False
        params.minConvexity = 0.87

        params.filterByInertia = False
        params.minInertiaRatio = 0.01

        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            detector = cv2.SimpleBlobDetector(params)
        else:
            detector = cv2.SimpleBlobDetector_create(params)  # default olarak bu kalabilir daha sonraki versiyonda.

        # Detect blobs.
        keypoints = detector.detect(im)
        k = keypoints
        # print(k)
        # print(type(k))
        print("Obje sayısı = ", len(k))

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
        # the size of the circle corresponds to the size of blob

        im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (10, 255, 0),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)    #BU GÜZEL
        # im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)  #BU DA İYİ
        # im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),  cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG) ÇALIŞMIYOR
        # im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)



        # Show blobs
        # cv2.imshow("Keypoints", im_with_keypoints)

        blobList = []
        blobList.append(im_with_keypoints)
        for i in range(len(k)):
            #print("X=", keypoints[i].pt[0], " Y=", keypoints[i].pt[1], "Cap=", (keypoints[i].size * mmpx))
            blobList.append(keypoints[i].size)
            # detector.detectAndCompute()
        #return blobList
        if(len(k)>0):
            x = int(keypoints[i].pt[0])
            y = int(keypoints[i].pt[1])
            r = int(keypoints[i].size/2)
            o = len(k)

            return im_with_keypoints, x, y, r, o
        return im_with_keypoints, 0, 0, 0, 0

    @classmethod    #blob analiz
    def ObjeAra(cls, source=0):
        time.sleep(0.5)
        img = Resim.TakePhoto(source)
        im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # im = cv2.imread("blob3.jpg", cv2.IMREAD_GRAYSCALE)

        params = cv2.SimpleBlobDetector_Params()
        params.minDistBetweenBlobs = 5

        params.minThreshold = 10 #10
        params.maxThreshold = 200#200

        params.filterByArea = True
        params.minArea = 30    #300- R=20px
        params.maxArea = 60    #500  R=25px

        params.filterByCircularity = False
        params.minCircularity = 0.01
        #params.maxCircularity = 0.8

        params.filterByConvexity = False
        params.minConvexity = 0.87

        params.filterByInertia = False
        params.minInertiaRatio = 0.01

        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            detector = cv2.SimpleBlobDetector(params)
        else:
            detector = cv2.SimpleBlobDetector_create(params)  # default olarak bu kalabilir daha sonraki versiyonda.

        # Detect blobs.
        keypoints = detector.detect(im)
        k = keypoints
        #print(k)
        #print(type(k))
        print("Obje sayısı = ", len(k))


        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
        # the size of the circle corresponds to the size of blob

        im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (10, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        #im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)    #BU GÜZEL
        # im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)  #BU DA İYİ
        # im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),  cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG) ÇALIŞMIYOR
        #im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        # Show blobs
        #cv2.imshow("Keypoints", im_with_keypoints)
        blobList = []
        blobList.append(im_with_keypoints)
        for i in range(len(k)):
            print("X=", keypoints[i].pt[0], " Y=", keypoints[i].pt[1], (keypoints[i].size * 0.0894))
            blobList.append(keypoints[i].size)
            # detector.detectAndCompute() bunu muhakkak çalış
        return blobList

    @classmethod ###
    def ShowImgPlot(cls, resim):
        #resimRGB = cv2.cvtColor(resim, cv2.COLOR_BGR2RGB)
        plt.imshow(resim)
        plt.show()

    @classmethod # otomatik renk tespitli olsun, tipe göre img döndürsün
    def CropImg(cls, img, x, y, w, h):
        if(img.shape[2]==2):
            #imgRGB = np.zeros(w, h, 3, 'uint8')
            imgRGB = img[y:y+h, x:x+w]
            return imgRGB
        else:
            #imgMono = np.zeros(w, h, 'uint8')
            imgMono = img[y:y+h, x:x+w]
            return imgMono

    @classmethod
    def OCRStream(cls, resim, dil = "tur"): #Aktif ayarlı fonksiyon
        RGB=True
        try:
            boyut = resim.shape[2]
            print(resim.shape)
            # print("renkli")
        except:
            # print("siyah beyaz")
            RGB = False

        if(RGB):
            gray = cv2.cvtColor(resim, cv2.COLOR_RGB2GRAY)
            gray = cv2.bitwise_not(gray)

        kernel = np.ones((2, 1), np.uint8)
        resim = cv2.erode(resim, kernel, iterations=1)
        resim = cv2.dilate(resim, kernel, iterations=1)

        # text = pytesseract.image_to_string(img, lang = "en")
        text = pytesseract.image_to_string(resim, lang = dil)
        return text

    @classmethod
    def PnGOzel(cls):
        #Fairy projesi için gerekli olan fonksiyonlar yer alacak.
        return 0

    @classmethod ###
    def KenarOlculeri(cls, resim, tip=0):   #tip=0 kenar ölçüleri, #tip=1 açı ölçüleri
        xMin = resim[1]
        xMiny = resim[2]
        xMax = resim[3]
        xMaxy = resim[4]
        yMinx = resim[5]
        yMin = resim[6]
        yMaxx = resim[7]
        yMax = resim[8]
        midX= (xMin+xMax)/2
        midY=(yMin+yMax)/2

        solOrt = (xMiny + yMin)/2
        sagOrt = (xMaxy + yMax)/2

        a1 = Resim.Vektor1D(xMin, xMiny, yMinx, yMin)
        a2 = Resim.Vektor1D(yMaxx, yMax, xMin, xMiny)

        lines = np.zeros((4, 1), float)
        mmpx = 0.0894
        #print("Köşeler= xMin={},{}  xMax={},{}  yMin={},{}    yMax={},{}".format(xMin, xMiny, xMax, xMaxy, yMinx, yMin, yMaxx, yMax))
        #if(xMaxy < xMiny):  #sağa dönüklük koşulu
        if(a2 < a1):#yMax'ın x'i eğerki x orta noktasının solundaysa, parça sağa dönüktür.
            print("Sağa dönük", a1, a2)  # Y ekseni aaşağı doğru artar(!)
            lines[0] = Resim.Vektor1D(xMin, xMiny, yMinx, yMin, mm1px=mmpx, detay=tip)# a1 kenar
            lines[1] = Resim.Vektor1D(yMinx, yMin, xMax, xMaxy, mm1px=mmpx, detay=tip)# a2 kenar
            lines[2] = Resim.Vektor1D(xMax, xMaxy, yMaxx, yMax, mm1px=mmpx, detay=tip)# a3 kenar
            lines[3] = Resim.Vektor1D(yMaxx, yMax, xMin, xMiny, mm1px=mmpx, detay=tip)# a4 kenar
        elif(a1 < a2):
            print("Sola dönük", a1, a2)
            lines[0] = Resim.Vektor1D(yMaxx, yMax, xMin, xMiny, mm1px=mmpx, detay=tip)# a1 açı
            lines[1] = Resim.Vektor1D(xMin, xMiny, yMinx, yMin, mm1px=mmpx, detay=tip)# a2 açı
            lines[2] = Resim.Vektor1D(yMinx, yMin, xMax, xMaxy, mm1px=mmpx, detay=tip)# a3 açı
            lines[3] = Resim.Vektor1D(xMax, xMaxy, yMaxx, yMax, mm1px=mmpx, detay=tip)# a4 açı

        return lines[0], lines[1], lines[2], lines[3]

    @classmethod
    def HatBelirle(cls,  resim):
        hat = cv2.findContours(resim, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hat = cv2.contour()
        return hat

    @classmethod    #3D döndürmek için faydalı
    def RotatePart(cls, resim):
        #Resim döndürme
        img = resim
        rows, cols, ch = img.shape
        print(img.shape)
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(img, M, (cols, rows))
        plt.subplot(121), plt.imshow(img), plt.title('Input')
        plt.subplot(122), plt.imshow(dst), plt.title('Output')
        plt.show()

    @classmethod ###
    def OrtRenkTonu(cls, obje):
        resim = obje[0]
        x = np.size(resim, 1)
        y = np.size(resim, 0)
        #print("Size*ort = {}x{}".format(x, y))
        ton = np.zeros((4), int)
        counter = 0
        tonM = 0
        for j in range(y):
            for i in range(x):
                ton[0] += resim[j, i][0]
                ton[1] += resim[j, i][1]
                ton[2] += resim[j, i][2]
                tonM += (int(resim[j, i][2]) + int(resim[j, i][1]) + int(resim[j, i][0])) // 3
                counter += 1
        ton[0] = ton[0] // counter
        ton[1] = ton[1] // counter
        ton[2] = ton[2] // counter
        ton[3] = tonM // counter
        #print("Ton=", ton)
        return ton

    @classmethod ###
    def RenkSkalaMax(cls, obje):
        resim = obje[0]
        x = np.size(resim, 1)
        y = np.size(resim, 0)
        #print("Size*max = {}x{}".format(x, y))
        maxSkala = np.zeros((4), int)
        bMax = 0
        gMax = 0
        rMax = 0
        mMax = 0
        for j in range(y):
            for i in range(x):
                if(resim[j, i][2] > bMax):
                    bMax = resim[j, i][2]

                if(resim[j, i][2] > gMax):
                    gMax = resim[j, i][1]

                if(resim[j, i][0] > rMax):
                    rMax = resim[j, i][0]

                m = (int(resim[j, i][2]) + int(resim[j, i][1]) + int(resim[j, i][0])) // 3
                if(m > mMax):
                    mMax = m

        maxSkala[0] = bMax
        maxSkala[1] = gMax
        maxSkala[2] = rMax
        maxSkala[3] = mMax

        return maxSkala

    @classmethod ###
    def RenkSkalaMin(cls, obje):
        resim = obje[0]
        x = np.size(resim, 1)
        y = np.size(resim, 0)
        #print("Size*min = {}x{}".format(x, y))
        minSkala = np.zeros((4), int)
        bMin = 255
        gMin = 255
        rMin = 255
        mMin = 255
        for j in range(y):
            for i in range(x):
                if (resim[j, i][2] < bMin):
                    bMin = resim[j, i][2]

                if (resim[j, i][2] < gMin):
                    gMin = resim[j, i][1]

                if (resim[j, i][0] < rMin):
                    rMin = resim[j, i][0]

                m = (int(resim[j, i][2]) + int(resim[j, i][1]) + int(resim[j, i][0])) // 3
                if (m < mMin):
                    mMin = m

        minSkala[0] = bMin
        minSkala[1] = gMin
        minSkala[2] = rMin
        minSkala[3] = mMin
        return minSkala

    @classmethod
    def GetInfo(cls, resim):
        x = np.size(resim, 1)
        y = np.size(resim, 0)
        c = np.size(resim[0, 0])
        d = "Null"

        if(c>1):
            C = "RGB"
            d = type(resim[0, 0][2])
        else:
            C = "MonoColor"
        E = "Boyut="+str(x) + "x" + str(y) + " Renk=" + C + " Data="+ str(d)
        return  E

    @classmethod ###
    def VektorHesapla(cls, p0x, p0y, p1x, p1y, mm1px=1.000, detay=0):# detay =0 uzunluk, 1= uzunluk ve açı(derece), 2= uzunluk(mm), açı, x ve y uzunluk(mm)
        x = p1x - p0x
        y = p1y - p0y
        #print("({},{})".format(x, y))
        hipotenus = math.sqrt(math.pow(x, 2)+(math.pow(y, 2)))

        aciR = math.acos(x/hipotenus)
        aciD = math.degrees(aciR)
        if(y > 0):
            aciD = 360 - aciD

        hipotenusMm = hipotenus * mm1px
        if(detay==0):
            return hipotenusMm
        elif(detay==1):
            return hipotenusMm, aciD
        elif(detay==2):
            return hipotenusMm, aciD, x, y


    @classmethod
    def Calibration(cls, mmPx, camX, camY, robX, robY, pX, pY):
        cRx = robX - (camX*mmPx)
        cRy = robY - (camY*mmPx)

        return (cRx + pX*mmPx), (cRy + pY*mmPx)

    @classmethod
    def SetPickPoint(cls, mX, mY, dX, dY, aci):


        return 0
    @classmethod
    def LineHistogram(self,resim, p0, p1, m, yon=True, rgb=True):# yon girilmezse default yatay alınır. m= yataydaki hizası
        repos=[]
        if(yon):   #False ise dikey
            if(rgb):
                reposR = []
                reposG = []
                reposB = []
                for i in range(p0, p1):
                    reposR.append(resim[m,i][2])
                    reposG.append(resim[m,i][1])
                    reposB.append(resim[m,i][0])
                plt.plot(reposR, 'red')
                plt.plot(reposG,'green')
                plt.plot(reposB, 'blue')
                plt.show()
                Resim.ShowImg(resim)

            else:
                for i in range(p0,p1):
                    repos.append((resim[m,i][2]+resim[m,i][1]+resim[m,i][0])//3)
                plt.plot(repos)
                plt.show()
                Resim.ShowImg(resim)

    @classmethod
    def Olcekle(cls, frame, oran):

        # frame= gelen resim, resim=yeni resim, yeni resim piksel= resimPx
        a = np.size(frame)
        x = np.size(frame, 0)  # x ekseni frame de 2. eksene yazılmış
        y = np.size(frame, 1)  # y ekseni frame de 1. eksene yazılmış

        start = time.time()
        print(time.time() - start)
        print(type(frame))
        print("Mevcut boyut={}x{}".format(x, y))
        xY = x // oran
        yY = y // oran
        resimPx = np.zeros((xY, yY, 3), 'uint8')
        print("Yeni boyut={}x{}".format(np.size(resimPx, 0), np.size(resimPx, 1)))
        yP = 0
        for j in range(0, y - 1):
            xP = 0
            for i in range(0, x - oran, oran):
                rBuf0 = 0
                gBuf0 = 0
                bBuf0 = 0
                for b in range(0, oran - 1):
                    rBuf0 += frame[i + b, j][0]
                    gBuf0 += frame[i + b, j][1]
                    bBuf0 += frame[i + b, j][2]
                rBuf1 = rBuf0 // oran
                gBuf1 = gBuf0 // oran
                bBuf1 = bBuf0 // oran
                xP += 1
                resimPx[xP, yP] = (rBuf1, gBuf1, bBuf1)
            if (j % oran == 0) and j > 0:
                yP += 1
        print(time.time() - start)
        return resimPx  #olcekli resim
        # cv2.imshow("Olcek=" + str(oran) + " Boyut=" + str(xY) + "x" + str(yY), resimPx)

    @classmethod
    def PeakPoints(cls, resim):
        x=np.size(resim, 1)
        y=np.size(resim, 0)
        start=time.time()
        r=0
        g=0
        b=0
        for i in range(y-1):
            for j in range(x-1):
                r += resim[i, j][2]
                g += resim[i, j][1]
                b += resim[i, j][0]
        r = r//(i*j)
        g = g//(i*j)
        b = b//(i*j)
        print(r, g, b)
        print("İslem suresi={}".format(time.time()-start))

    @classmethod
    def HariciResim(cls, path):
        resim = cv2.imread(path)
        return resim

    @classmethod ###
    def DetectEdges(cls, resim, t1=100, t2=100, edges=5, aptSize=3, L2g=False):#buraya basitçe threshold değeri verebilirim!
        return cv2.Canny(resim, t1, t2, edges, aptSize, L2gradient=L2g)

    @classmethod
    def DetectLines(cls, img, tMin=100, tMax=100, minLinePx=0, edges=5, aptSize=3, L2g=False):
        edges = Resim.DetectEdges(img)
        whiteLines = np.zeros((np.size(img, 0), np.size(img, 1), 1), 'uint8')
        minLineLength = 30  # 30
        maxLineGap = 10  # 10
        lines = 15
        # lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength,maxLineGap)
        # cizgiler = cv2.HoughLinesP(edges, 1, np.pi/180, 15, minLineLength, maxLineGap)
        cizgiler = cv2.HoughLinesP(edges, 1, np.pi / 180, 15, lines, minLineLength, maxLineGap)
        cizgiVar=False
        try:
            print(len(cizgiler))
            # print(cizgiler)

            for x in range(0, len(cizgiler)):
                for x1, y1, x2, y2 in cizgiler[x]:
                    cv2.line(whiteLines, (x1, y1), (x2, y2), (255), 1)

            # Resim.ShowImg(img)  # cizgi eklenmis orjinal img
            #return whiteLines yüz tespitinde netlik seviyesi için deaktif edildi.
            return whiteLines.shape
        except:
            print("Çizgi bulunamadi")
            return 0

    @classmethod    #----------------------------------------------     TANIM YAZILACAK
    def DetectLinesV2(cls, img):

        return None

    @classmethod    #----------------------------------------------     TANIM YAZILACAK
    def FindCircles(cls, resim, rMin=20, rMax=40, minConvex=0, maxConvex=1, minInertia=0, maxInertia=1):

        pass

    @classmethod
    def DetectCorners(cls, resim):
        img = resim
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
        corners = np.int0(corners)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(img, (x, y), 3, 255, -1)
        #plt.imshow(img), plt.show()
        return img

    @classmethod ### source Set ayarlanacak
    def LiveStream(cls, source, type=0, olcek=100, x=640, y=480):#source= görüntüleme kaynağı, type= fonksiyon(0=normal görüntü,1=edge detect,2=...)
        cap = cv2.VideoCapture(source)
        if x != 640 or y != 480:
            cap.set(3, x)
            cap.set(4, y)
        exposure = 0.0
        start = 0
        while True:
            #time.sleep(0.05)
            exposure = (time.time() - start)*1000
            #print(" Exposure Time= " + str(exposure) + "ms")
            start = time.time()
            ret, img = cap.read()
            if(olcek > 1):
                print("Olcek=", olcek)
                #img = cls.Olcekle(img, olcek)
                img = Resim.Resize(img, olcek)
            if(type == 1):# Çizgi tespiti
                img = cls.DetectEdges(img, 100, 100, 3, 3, True)
                cv2.imshow("Cizgiler + Scale=1/"+str(olcek), img)
            elif(type==0):
                cv2.imshow('Normal + Scale=1:'+str(olcek/100), img)
            if (cv2.waitKey(10) & 0xFF == ord('e')):
                cap.release()
                cv2.destroyAllWindows()
                break

    @classmethod
    def ReadPath(cls, path):
        resim = cv2.imread(path)
        return resim

    @classmethod
    def ReadBarcode(cls, img):
        barcodes = pyzbar.decode(img)
        for barcode in barcodes:
            # extract the bounding box location of the barcode and draw the
            # bounding box surrounding the barcode on the image
            (x, y, w, h) = barcode.rect
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            # the barcode data is a bytes object so if we want to draw it on
            # our output image we need to convert it to a string first
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type
            # draw the barcode data and barcode type on the image
            text = "{} ({})".format(barcodeData, barcodeType)
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)
            return

    @classmethod ### Buna görüntü kalitesi levelları ekle("mesela vga, hd, fhd,4k,8k gibi en sonda yine mevcuttaki custom ölçek olsun)
    def TakePhoto(cls, source, kaydet=False, isim="dateNow", x=640, y=480):
        foto = cv2.VideoCapture(source)
        if x != 640 or y != 480:
            foto.set(3, x)
            foto.set(4, y)
        ret, frame = foto.read()

        if (kaydet):
            if(isim=="dateNow"):
                tarih = datetime.datetime.today()
                ad ="Photos\Pathless\A"+ str(tarih.year) + str(tarih.month) + str(tarih.day) + str(tarih.hour) + str(tarih.minute)
                print(ad)
                cls.Kaydet(ad+".jpg", frame)
            else:
                cls.Kaydet(isim, frame)

        return frame

    @classmethod
    def CaptureFromVideo(cls, vidcap, sec, pathName="test.jpg"):
        sec = round(sec, 2)
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        # vidcap.get(cv2.CAP_PROP_SPEED)
        frame, image = vidcap.read()
        return image

    @classmethod ###
    def AutoLearn(cls, frame, frameRGB, catch=True, RGB=True, roi=False, isaretli=False, xS=0, xE=1, yS=0, yE=1):    # Noktaları yakala, roi eklenirse o alandakileri al. yoksa tüm frame
        #AutoLearn gerekli olan en uç bilgilere kadar işleyebilmeli ve nesne künyesine yazmalı.
        if not(roi):
            x = np.size(frame, 1)
            y = np.size(frame, 0)   #xMin, xMiny, xMax, xMaxy, yMin, yMinx, yMax, yMaxx #toplam 9 değişken
            xMin  = x
            xMiny = 0
            xMax  = 0
            xMaxy = 0
            yMin  = y
            yMinx = 0
            yMax  = 0
            yMaxx = 0
            dotCounter = 0
            for j in range(y-1):
                for i in range(x-1):
                    if(frame[j,i] > 250):
                        #print(i,j,"  White dot")
                        dotCounter += 1
                        if(i < xMin):
                            xMin = i
                            xMiny = j
                        if(i > xMax):
                            xMax = i
                            xMaxy = j
                        if(j < yMin):
                            yMin = j
                            yMinx = i
                        if(j > yMax):
                            yMax = j
                            yMaxx = i

            autoROI = print("AutoLearn ile bulunan obje boyutu: \n"
                  "({},{})----------({},{})\n   |               |\n   |               |\n   |               |\n   |               |\n({},{})---------({},{})".format(xMin,yMin,xMax,yMin,xMin,yMax,xMax,yMax))

            if(catch):
                if(RGB):
                    #yeni rgb array
                    lineX = xMax - xMin
                    lineY = yMax - yMin
                    nesneRGB = np.zeros((lineY, lineX, 3), 'uint8')
                    for j in range(yMin, yMax):
                        for i in range(xMin, xMax):
                            if(isaretli):
                                if(frame[j, i]>250 ):
                                    nesneRGB[j - yMin, i - xMin][0] = 0
                                    nesneRGB[j - yMin, i - xMin][1] = 255
                                    nesneRGB[j - yMin, i - xMin][2] = 0
                                else:
                                    nesneRGB[j - yMin, i - xMin][2] = frameRGB[j, i][2]
                                    nesneRGB[j - yMin, i - xMin][1] = frameRGB[j, i][1]
                                    nesneRGB[j - yMin, i - xMin][0] = frameRGB[j, i][0]
                            else:
                                nesneRGB[j - yMin, i - xMin][2] = frameRGB[j, i][2]
                                nesneRGB[j - yMin, i - xMin][1] = frameRGB[j, i][1]
                                nesneRGB[j - yMin, i - xMin][0] = frameRGB[j, i][0]

                    #print(dotCounter)
                    #print(autoROI)
                    return nesneRGB, xMin, xMiny, xMax, xMaxy, yMinx, yMin, yMaxx, yMax #toplam 9 değişken     # ***EN GENEL RETURN DURUMU***
                else:
                    #yeni monocolor array
                    xLine = xMax - xMin
                    yLine = yMax - yMin
                    nesne = np.zeros((yLine, xLine, 1), 'uint8')
                    for j in range(yMin, yMax-1):
                        for i in range(xMin, xMax-1):
                            nesne[j-yMin, i-xMin] = frame[j, i]
                    #print(dotCounter)
                    print(autoROI)
                    return nesne, xMin, xMiny, xMax, xMaxy, yMin, yMinx, yMaxx, yMax #toplam 9 değişken

            #print("Köşeler=> x0={},x1={},y0={},y1={}".format(xMin,xMax,yMin,yMax))
            print(autoROI)
            return xMin, xMax, yMin, yMax

    @classmethod
    def BGRtoGray(cls, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    @classmethod
    def CVTColor(cls, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    @classmethod
    def AddText(cls, img, text, x=0, y=35, size=2, scale = 1, thickness = 2):
        font = cv2.FONT_ITALIC
        RGB = True
        try:
            boyut = img.shape[2]
            # print("renkli")
        except:
            # print("siyah beyaz")
            RGB = False

        if(RGB):
            cv2.putText(img, text, (x, y), font, scale, (255, 0, 0), size, cv2.LINE_AA)
        else:
            cv2.putText(img, text, (x, y), font, 1, (255), size, cv2.LINE_AA)
        return img

#----------------DATABASE İŞLEMLERİ----------------------
    @classmethod
    def DatabaseInit(cls):
        return 0

    @classmethod
    def SqlFotoKaydet(cls, database, img, id):
        return 0

    @classmethod
    def SqlFotoCek(cls, database, id):
        return 0

    @classmethod
    def SqlGetToolSettings(cls, tool, id):
        return 0

    @classmethod
    def SqlSetToolSettings(cls, tool, id):
        return 00000000

#----------------DATABASE İŞLEMLERİ----------------------

    @classmethod
    def BaslerPhoto(cls, init, gain=12, exposure=30000):
        if (init):
            camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            camera.Open()

            #Basler Settings
            camera.ExposureAuto.SetValue('Off')  # exposure ayarları
            camera.ExposureTime.SetValue(exposure)  # microsecond

            camera.GainAuto.SetValue('Off')  # gain ayarları
            camera.Gain.SetValue(gain)

            camera.Width.Value = camera.Width.Max  # pixel frame ayarları
            camera.Height.Value = camera.Height.Max

            # camera.Close()

            print("DeviceClass: ", camera.GetDeviceInfo().GetDeviceClass())
            print("DeviceFactory: ", camera.GetDeviceInfo().GetDeviceFactory())
            print("ModelName: ", camera.GetDeviceInfo().GetModelName())

            Hardware_Trigger = False

            if Hardware_Trigger:  # reset registration
                camera.RegisterConfiguration(pylon.ConfigurationEventHandler(), pylon.RegistrationMode_ReplaceAll,
                                             pylon.Cleanup_Delete)
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            converter = pylon.ImageFormatConverter()
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grabResult.GrabSucceeded():
                # Access the image data
                image = converter.Convert(grabResult)
                img = image.GetArray()
                # cv2.imwrite('save_images/%06d.png' % count, img)

                # OCR TEST
                """img = Resim.Resize(img, yuzde=20)
                text = Resim.OCRStream(img)
                print("OUTPUT=", text)
                #img = Resim.AddText(img, text=text)"""
                # img = Resim.Resize(img,485)
                img = Resim.AddText(img, "unitek")
                return img
        else:
            Hardware_Trigger = False

            if Hardware_Trigger:  # reset registration
                camera.RegisterConfiguration(pylon.ConfigurationEventHandler(), pylon.RegistrationMode_ReplaceAll,
                                             pylon.Cleanup_Delete)
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            converter = pylon.ImageFormatConverter()
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grabResult.GrabSucceeded():
                # Access the image data
                image = converter.Convert(grabResult)
                img = image.GetArray()
                # cv2.imwrite('save_images/%06d.png' % count, img)

                # OCR TEST
                """img = Resim.Resize(img, yuzde=20)
                text = Resim.OCRStream(img)
                print("OUTPUT=", text)
                #img = Resim.AddText(img, text=text)"""
                # img = Resim.Resize(img,485)
                img = Resim.AddText(img, "unitek")
                return img

    @classmethod
    def DrawLine(cls, img, x1,y1,x2,y2, c=1, d=2, color=(255, 255, 255)):
        #c==1 ise renkli, değilse grayscale
        if(c==1):
            img = cv2.line(img, (x1, y1), (x2, y2), color, 2*d)
        else:
            img = cv2.line(img, (x1, y1), (x2, y2), (255), 2*d)
            img = cv2.line(img, (x1, y1), (x2, y2), (0), d)
        return img

    @classmethod
    def LearnPartContour(cls,img, limitAlan=500):           #-----------------------BURADA KALDIN KONTÜR TESPİTİ PARÇA ÖĞRENME-----------------
        contourData = []
        # -----Kontür Tespit Parametreleri------
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, thresh=130, maxval=255, type=0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #-----Kontür Kenardan Ayırma ve Tekleme-------------
        if(len(contours)>1):
            print(len(contours)-1, "adet kontür tespit edildi")
        cnt = contours[1]
        for i in range(0, len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            area = cv2.contourArea(contours[i])
            #-----DEBUG Amaçlı görselleştirme--------
            cv2.drawContours(img, [cnt], contourIdx=0, color=(0, 255, 0),
                             thickness=2)  # contourIdx=-1 hepsi, contourIdx=0,1,2,3... spesifik contour

            Resim.AddText(img, text=str(i), x=x + 30, y=y + 30)
            if(x>10 ):
                if(area>limitAlan):
                    cnt = contours[i]
                    print("Tespit edilen", i, "-nolu kontür şartlara uyuyor.")


                    massCenter = cv2.moments(cnt)
                    cX = int(massCenter['m10'] / massCenter['m00'])
                    cY = int(massCenter['m01'] / massCenter['m00'])

                    contourData.append(cnt)
                    contourData.append(area)
                    contourData.append()




                else:
                    print("Tespit edilen", i, "-nolu kontür alan şartına uymuyor")
            else:
                print("Tespit edilen", i, "-nolu kontür konum şartına uymuyor")




        return 0




    @classmethod
    def DrawContour(cls, img: object, threshMin: object = 90, threshMax: object = 255, ) -> object:
        #img = Resim.CropImg(img, 5,5,img.shape[1]-5, img.shape[0]-5)
        #img = Resim.Blur(img, k=9, type=2)
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, threshMin, threshMax, type=0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            # cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
                            # cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        cnt = contours[-1]
        #print(cnt)
        cv2.drawContours(img, [cnt], contourIdx=0, color=(50, 255, 0), thickness=2)     #contourIdx=-1 hepsi, contourIdx=0,1,2,3... spesifik contour

        return img, contours, cnt


    @classmethod
    def DrawRectangle(cls, img, x, y, w, h, thickness=1):
        img = cv2.rectangle(img, (x, y), (x+w, y+h), thickness=thickness)
        return img

    @classmethod
    def DrawCircle(cls, img, x, y, r, c=1, cM=100):
        #eğer c=1 se (default) renkli gibi işlem görür, değilse mono color beyaz içinde siyah olur
        if(c==1):
            img = cv2.circle(img, (x, y), r, (200, 150, 60), -1)
        else:
            #img = cv2.circle(img, (x, y), r+1, (255), -1)
            img = cv2.circle(img, (x, y), r, (cM), -1)
        return img

    @staticmethod
    @jit()
    def CornerDetectionMono(img, arrow, lim=75, type=0, gap=10):  # monocolor image için corner detection
        # type=0 ise siyahtan beyaza veya beyazdan siyaha, type==1 beyazdan siyaha, type==2 siyahtan beyaza
        # düzlem denklemi üzerine x iterasyonundan alınan y değerindeki mono color limitlerine göre corner tespiti
        corners = list()
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        x1 = arrow[0]
        y1 = arrow[1]
        x2 = arrow[2]
        y2 = arrow[3]
        dY = y2 - y1
        dX = x2 - x1
        # print(dY)
        m = dY / dX
        x = 0
        for x in range(x1, x2):  # x1,x2
            y = int(m * (x - x1)) + y1
            b = (y, x)
            y2 = int(m * ((x + 1) - x1)) + y1
            a = (y2, x + 1)  # yazımı ters

            # v = img[b][0] - img[a][0]
            vB = img[b][0]
            vA = img[a][0]
            v = int(vB) - int(vA)
            v = abs(v)
            print(v)
            if v > lim:
                x = x + 5
                # print("v=", v, "---", vB, vA)
                corners.append(a)

        for i in corners:
            img = Resim.DrawCircle(img, i[1], i[0], r=5)
            # print(img[a])
        # print(corners)
        return img, corners  # x,y
        # return img

    @staticmethod
    @jit()
    def CornerDetectionMono(img, arrow, lim=150, type=0):  # monocolor image için corner detection
        # type=0 ise siyahtan beyaza veya beyazdan siyaha, type==1 beyazdan siyaha, type==2 siyahtan beyaza
        # düzlem denklemi üzerine x iterasyonundan alınan y değerindeki mono color limitlerine göre corner tespiti
        corners = list()
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        x1 = arrow[0]
        y1 = arrow[1]
        x2 = arrow[2]
        y2 = arrow[3]
        dY = y2 - y1
        dX = x2 - x1
        # print(dY)
        m = dY / dX
        for x in range(arrow[0], arrow[2]):
            y = int(m * (x - x1)) + y1
            b = (x, y)
            y2 = int(m * ((x + 1) - x1)) + y1
            a = (x + 1, y2)
            # img.shape = 1000,1700

            if (img[x, y][0] < 200):
                img = Resim.DrawCircle(img, x, y, r=2)
                p = img[x - 1, y - 1][0]
                print(p)
            # img = Resim.DrawCircle(img, x, y, r=1, c=0, cM=50)
            # print(img[x,y])
            # Resim.DrawCircle(img, 750, 460, r=15)

            # Resim.ShowImgPlot(img)
            """if(img[x,y][0]+img[x,y][1]+img[x,y][2]) < 450:
                img = Resim.DrawCircle(img, x, y, r=5, c=1, cM=50)"""
        return img  # x,y

    @staticmethod
    @jit()
    def IntersectionDetect(img, arrow, line):
        img = Resim.DrawLine(img, arrow[0], arrow[1], arrow[2], arrow[3],d=1) # Kesişim tespitleme işareti
        img = Resim.DrawCircle(img, arrow[2], arrow[3], r=6)
        #img = Resim.DrawLine(img, line[0], line[1], line[2], line[3]) # Kesişim kenarı işareti

        #crossingDot = np.zeros(2, 'uint8')
        # arrow, line = np.ndarray(x,y),2
        return img


    @classmethod
    def FindParts(cls, alan, obje, match=0.8): #non rotational
        RGB = True
        try:
            boyut = alan.shape[2]
            #print("renkli")
        except:
            #print("siyah beyaz")
            RGB = False

        if(RGB):
            img_rgb = alan
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)    # alan
        else:
            img_gray = alan

        pt_List = []

        #template = cv2.cvtColor(obje, cv2.COLOR_BGR2GRAY) #obje
        template = obje
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        matchRate = match
        loc = np.where(res >= matchRate)    #Match içerikleri burada
        #print(type(res[0]))
        #print(loc[0])
        parts = 0
        x=0
        y=0
        for pt in zip(*loc[::-1]):
            if(RGB):
                cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (80, 220, 10), 1)
                x = pt[0]
                y = pt[1]
                pt_List.append(pt)
                parts += 1
            else:
                cv2.rectangle(img_gray, pt, (pt[0] + w, pt[1] + h), (255), 1)
                x = pt[0]
                y = pt[1]
                pt_List.append(pt)
                parts += 1

        if(parts>0):
            print("onaylı iterasyon=", parts)
            print("Onaylı iterasyon[0]= ", pt_List[0])

            #cv2.imshow('Detected',img_rgb)
            #Resim.ShowImgPlot(img_rgb)
            #print("IMG RGB SHAPE", img_rgb.shape)
            if(RGB):
                return img_rgb, x, y, w, h
            else:
                return img_gray, x, y, w, h
        else:
            #print("match yok")
            return alan, x, y, w, h

    @classmethod
    def Match(cls, alan, obje, match):
        return

    @classmethod
    def Resize(cls, img, yuzde, method=0):

        #img = cv2.imread('fairy_s_arka_inkjet.jpg', cv2.IMREAD_UNCHANGED)
        #print('Original Dimensions : ', img.shape)
        scale_percent = yuzde  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        if(method==0):
            # resize image
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        elif(method==1):
            # resize image
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_LANCZOS4)
        elif (method == 2):
            # resize image
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
        elif (method == 3):
            # resize image
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
        elif (method == 4):
            # resize image
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
        return resized

    @classmethod
    def NesneAyirici(cls, resim):
        im = cv2.imread("deneme000.jpg", cv2.IMREAD_GRAYSCALE)
        #im = obje
        im0 = resim
        detector = cv2.SimpleBlobDetector()
        keypoints = detector.detect(im)
        im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        #cv2.imshow("Keypoints", im_with_keypoints)
        #cv2.waitKey(0)
        return im_with_keypoints

    @classmethod
    def Kaydet(cls, resim, ad):
        cv2.imwrite(ad, resim)
        return True

    @classmethod
    def TakeVideo(cls, source, resLevel ):
        return True

    @classmethod
    def ReadVideo(cls, file):   # Video dosyalarının framelerini okur ve döndürür.
        cap = cv2.VideoCapture(file)
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                cv2.imshow('Frame', frame)
        return True

    @classmethod # OCR fonksiyonu basit düzey-çalışan
    def OCR(cls, img, lang="tur",):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.bitwise_not(gray)

        kernel = np.ones((2, 1), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
        img = cv2.dilate(img, kernel, iterations=1)

        if(lang=="tur"):
            text = pytesseract.image_to_string(img, lang="tur")
            if (text != ""):
                return text
            else:
                return "noText"

        elif(lang=="en"):
            text = pytesseract.image_to_string(img, lang="en")
            if(text != ""):
                return text
            else:
                return "noText"

    @classmethod
    def Info(cls):
        print("\n"
              "Mevcut Fonksiyonlar= \n"
              "1)  size(self)          -->return=np.ndarray size(int) \n"
              "2)  TakePhoto(source)   -->return=np.ndarray resim \n"
              "3)  ShowImg(resim)      -->return=ekran goruntusu, e=exit \n"
              "4)  LineHistogram(resim,baslangicNoktasi,bitisNoktasi,diger eksen kordinati,yon=true,rgb=true)-->return=Histogram ekranı \n"
              "5)  Olcekle(frame,oran) -->return=Olceklenmis np.ndarray resim \n"
              "6)  DetectEdges(resim,t1=100,t2=100,edges=5,aptSize=3,L2g=True) --->return=grayscale yakalanan çizgiler \n"
              "7)  LiveStream(source,type,olcek=0) -->return= while=True yayın ekranı,e=exit \n"
              "8)  PeakPoints(resim)   -->Print=Test değeri \n"
              "9)  AutoLearn(cls, frame, frameRGB, catch=False, RGB=False, roi=False, xS=0, xE=1, yS=0, yE=1)"
              ")  Info()              -->Print=Tüm fonksiyonlar \n")

#-------------SAMPLES----------------------
