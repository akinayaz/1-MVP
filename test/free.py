#free
import cv2, numpy as np, math
#from libs.GoruntuIsleme import Resim

def VektorHesapla( p0x, p0y, p1x, p1y, mm1px=1.000, detay=0):# detay =0 uzunluk, 1= uzunluk ve açı(derece), 2= uzunluk(mm), açı, x ve y uzunluk(mm)
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

arrow_img_path = "assets/inputs/images/arrow.png"
arrows_img_path = "assets/inputs/images/arrows.png"

### 1---Obje Öğrenme:
     
# -----Öğretilen Parça Parametreleri---
obje = cv2.imread(arrow_img_path)# robot.png
objeKunye = np.zeros(10, dtype=float)
#-----Kontür Tespit Parametreleri------
objeGray = cv2.cvtColor(obje, cv2.COLOR_BGR2GRAY)
##Deneme 
#cv2.imshow("test", objeGray)
#cv2.waitKey(0)
##Deneme

ret, objeThresh = cv2.threshold(objeGray, thresh=130, maxval=255, type=0)
objeContours, objeHierarchy = cv2.findContours(objeThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
objeAlan = 0
cntObj = 0
for i in range(0, len(objeContours)):
    cntObje = objeContours[i]# buradan contour seçimi yap
    x, y, w, h = cv2.boundingRect(cntObje)
    alanX = cv2.contourArea(cntObje)
    #---- Kontür Eliminasyon ve Tespit -----------
    if(x > 1):#50#başlangıç konumu.(bu ileride mouse ile seçilebilir.)
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
d0, aci0 = VektorHesapla(p0x=cXO, p0y=cYO, p1x=box[0][0], p1y=box[0][1], detay=1)
d1, aci1 = VektorHesapla(p0x=cXO, p0y=cYO, p1x=box[1][0], p1y=box[1][1], detay=1)
d2, aci2 = VektorHesapla(p0x=cXO, p0y=cYO, p1x=box[2][0], p1y=box[2][1], detay=1)
d3, aci3 = VektorHesapla(p0x=cXO, p0y=cYO, p1x=box[3][0], p1y=box[3][1], detay=1)
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
cv2.imshow("test", obje)
cv2.waitKey(0)

##----- CANLI KAMERA SABİTLERİ------------
objelerKunye = np.zeros((10, 10), dtype=float)

#2--------

