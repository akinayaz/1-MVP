#Contour Based Matching
#Objelerin kontürlerini ve obje künyelerini kullanır.
#Objelerin künyesinin oluşturulması için kullanılan künyeleme programı fonksiyonları.

import cv2, numpy as np, time, math

#region Definition:
mmPx = 0.1122
pickAci = 270
pickL = 15 #mm cinsinden
#Camera ve robotun kalibrasyon yapıldığı noktalar değeri
cCX = 996.308 #Camera Calibration X
cCY = 576.539
rCX = 126.33 #Robot Calibration X
rCY = 73.22
#endregion

#region Tool Fonksiyonları

def DrawCircle(img, x, y, r, c=1, cM=100):
        #eğer c=1 se (default) renkli gibi işlem görür, değilse mono color beyaz içinde siyah olur
        if(c==1):
            img = cv2.circle(img, (x, y), r, (200, 150, 60), -1)
        else:
            #img = cv2.circle(img, (x, y), r+1, (255), -1)
            img = cv2.circle(img, (x, y), r, (cM), -1)
        return img

def AddText(img, text, x=0, y=35, size=2, scale = 1, thickness = 2):
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

def VektorHesapla(p0x, p0y, p1x, p1y, mm1px=1.000, detay=0):# detay =0 uzunluk, 1= uzunluk ve açı(derece), 2= uzunluk(mm), açı, x ve y uzunluk(mm)
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

def Resize(img, yuzde, method=0):

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

def DrawLine(img, x1,y1,x2,y2, c=1, d=2, color=(255, 255, 255)):
        #c==1 ise renkli, değilse grayscale
        if(c==1):
            img = cv2.line(img, (x1, y1), (x2, y2), color, 2*d)
        else:
            img = cv2.line(img, (x1, y1), (x2, y2), (255), 2*d)
            img = cv2.line(img, (x1, y1), (x2, y2), (0), d)
        return img

def PointFromVector(img, pX, pY, aci, boy=50, type=1):
        aciRad = math.radians(aci+90)
        sin = math.cos(aciRad)
        cos = math.sin(aciRad)
        resX = round(pX + (cos*boy))
        resY = round(pY + (sin*boy))
        if(type==0):
            return resX, resY
        if(type==1):
            img = DrawCircle(img, resX, resY, r=5)
            return img, resX, resY
        if(type==2):
            img = DrawLine(img, pX, pY, resX, resY)
            return img, resX, resY

def Calibration(mmPx, camX, camY, robX, robY, pX, pY):
        cRx = robX - (camX*mmPx)
        cRy = robY - (camY*mmPx)

        return (cRx + pX*mmPx), (cRy + pY*mmPx)


#endregion

def GenObjectTags(obje, params=0):
    """" Contour Match için Obje Künyesi oluşturma fonksiyonu.
    Obje ve bazı tespit parametreleri alır.
    Künye list'ini döndürür."""
    objeKunye = np.zeros(10, dtype=float)
    #-----Kontür Tespit Parametreleri------
    objeGray = cv2.cvtColor(obje, cv2.COLOR_BGR2GRAY)
    
    ret, objeThresh = cv2.threshold(objeGray, thresh=130, maxval=255, type=0)
    if not (ret):
        print("objeThresh ret = False, threshold alınamadı. Return False!")
        return False
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
    
        #----Kontür Ağırlık Merkezi-------
    massCenter = cv2.moments(cntObje)
    cXO = int(massCenter['m10']/massCenter['m00'])
    cYO = int(massCenter['m01'] / massCenter['m00'])
    obje = DrawCircle(obje, x=cXO, y=cYO, r=5)

    #----Kontür Açısal Boundary------
    rect = cv2.minAreaRect(cntObje)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

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
    #cv2.imshow("test", obje)
    #cv2.waitKey(0)
    return objeKunye, obje

def FindContourMatch(img, objeKunye, matchRate = 0.6):
    matchList = [] #pX,pY,açı,matchRate
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
                        d0, aci0 = VektorHesapla(p0x=cX, p0y=cY, p1x=box[0][0], p1y=box[0][1], detay=1)
                        d1, aci1 = VektorHesapla(p0x=cX, p0y=cY, p1x=box[1][0], p1y=box[1][1], detay=1)
                        d2, aci2 = VektorHesapla(p0x=cX, p0y=cY, p1x=box[2][0], p1y=box[2][1], detay=1)
                        d3, aci3 = VektorHesapla(p0x=cX, p0y=cY, p1x=box[3][0], p1y=box[3][1], detay=1)
                        obj1 = (d0, d1, d2, d3)
                        obj1Aci = (aci0, aci1, aci2, aci3)
                        durum, rate, sira = ContourAxisMatch(obj0, obj1, limit=matchRate)
                        print(durum, rate, sira)
                        if(durum):
                            cv2.drawContours(img, [cnt], contourIdx=0, color=(0, 255, 0),
                                         thickness=2)
                            img = cv2.drawContours(img, [box], 0, (255, 255, 0), 2)
                            AddText(img, text=str(round(100-rate, 2)), x=box[(sira+2) % 4][0], y=box[(sira+2) % 4][1],)
                            img = DrawCircle(img, x=cX, y=cY, r=10)
                            aci = obj1Aci[sira] - objeKunye[6]
                            if(aci<0):
                                aci = 360+aci
                            # ----Kontür Pick Point-------------
                            img, cX, cY = PointFromVector(img, cX, cY, aci=(pickAci+aci), boy=(pickL/mmPx))# piksel cinsinden
                            cXR = cX
                            cYR = img.shape[0] - cY
                            AddText(img, text=str(round(aci, 2)), x=box[(sira + 2) % 4][0],
                                          y=box[(sira + 2) % 4][1]+40)
                            img, xi, yi = PointFromVector(img, cX, cY, aci=aci-89, boy=160, type=2)
                            img, xi, yi = PointFromVector(img, cX, cY, aci=aci+1, boy=50, type=2)
                            AddText(img, text="("+str(partNo)+")", x=box[(sira + 2) % 4][0],
                                          y=box[(sira + 2) % 4][1]-40)
                            AddText(img, text=str(cXR) +"x"+ str(cYR), x=box[(sira + 2) % 4][0],
                                          y=box[(sira + 2) % 4][1] + 70)
                            rX, rY = Calibration(mmPx, cCX, cCY, rCX, rCY, cXR, cYR)
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
                            #newSocket.send(strRobot)
                            print("i=",i)
                            #break

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
        #Experimental:
        return img
    except:
        print("Bir hata oluştu...")

if __name__ == "__main__":
    #Code:
    objPath = "assets/inputs/images/arrow.png"
    imagePath = "assets/inputs/images/arrows.png"
    #-------------------
    objImage = cv2.imread(objPath)
    tags, objeVisual = GenObjectTags(objImage)
    image = cv2.imread(imagePath)
    image = FindContourMatch(image,tags)

    cv2.imshow("matching", image)
    cv2.waitKey(0)

    print(tags)
