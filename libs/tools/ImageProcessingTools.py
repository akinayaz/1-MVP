# Basit Image Processing Tool Fonksiyonları #
# Sağ alt->Python 3.9.13('mvpEnv':venv)

"""
1- Resize Image
2- Crop Image
3- Rotate Image
4-  
"""

"""Hazırlanan Fonksiyonlar:
1-SplitRGB
2-Resize
3-Crop
4-Rotate
"""

"""Sources:
Rotate: https://medium.com/analytics-vidhya/rotating-images-with-opencv-and-imutils-99801cb4e03e

"""
import cv2
import numpy as np
import imutils

def SplitCh(img, ch):#* ikincil.
    """BGR görüntünün kanallarını ayırıp döndüren fonk.
    """
    #(B, G, R) = cv2.split(image)
    #img = cv2.cvtColor(R, cv2.COLOR_GRAY2RGB)
    #image[:,:,0]=255
    image[:,:,1]=0
    image[:,:,2]=0
    #print(R.shape)
    return image

def SplitRGB(img, ch): #* birincil
    (B, G, R) = cv2.split(img)
    #img = cv2.cvtColor(R, cv2.COLOR_GRAY2RGB)
    if(ch=="R"):
        return R
    elif(ch=="G"):
        return B
    elif(ch=="B"):
        return B
    else:
        return False
    return R

def Resize(img, scale, interpolation=0): 
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dim = (width, height)
    #resized image
    if(interpolation==0):
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)# OpenCV default
    elif(interpolation==1):
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)# Görüntü shrinklemek için
    elif(interpolation==2):
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)# En yavaşı fakat en verimlisi
    return resized

def Crop(img, p1, p2):
    """Img üzerinden p1(y1,x1) ve p2(y2,x2) noktaları arasındaki alanı keserek döndürür.
    img[y1->y2, x1->x2]"""
    cropImg = img[ p1[0] : p2[0], p1[1] : p2[1]  ]
    return cropImg

def RotateCV(img, deg, type=0):
    """Resmi döndürme ve döndürürken ekranda konumlandırma fonksiyonu.
     Koordinat sistemi + yönüne göredir, saat yönünün tersine döndürür. """
    # get the dimensions of the image and calculate the center of the
    # image
    height, width = img.shape[:2]
    centerX, centerY = (width // 2, height // 2)
    # # rotate our image by 45 degrees around the center of the image

    # get rotation matrix
    M = cv2.getRotationMatrix2D((centerX, centerY), deg, 1.0)

    # rotate image
    rotated = cv2.warpAffine(img, M, (width, height))
    return rotated

def Rotate(img, deg, bound=True):
    if(bound):
        rotated = imutils.rotate_bound(image, deg)
    else:
        rotated = imutils.rotate(img, deg)
    return rotated

if __name__ == "__main__":
	# Code:
    image = cv2.imread("assets/inputs/images/colors.png") # assets/inputs/images/robotlar.png
    # Display the image
    #image = Resize(image, 30,interpolation=0)
    #image = Crop(image, (0,0),(500,500))
    #image = Rotate(image, 80)
    #image = SplitRGB(image, ch="R")

    print(image.shape)
    cv2.imshow("Image", image)
    # Wait for the user to press a key
    cv2.waitKey(0)
    # Close all windows
    cv2.destroyAllWindows()
    