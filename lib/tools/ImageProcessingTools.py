# Basit Image Processing Tool Fonksiyonları #
#Python 3.10.9-64b-bit LF UTF8
"""
1- Resize Image
2- Crop Image
3- Rotate Image
4-  
"""

"""Sources:
Rotate: https://medium.com/analytics-vidhya/rotating-images-with-opencv-and-imutils-99801cb4e03e


"""
import cv2
import numpy as np



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

def Rotate(img, deg, type=0):
    """Resmi döndürme ve döndürürken ekranda konumlandırma fonks """
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

def RotateImu(img, deg=45):
    rotated = imutils.rotate(image, -30)


if __name__ == "__main__":
	# Code:
    image = cv2.imread("assets/inputs/images/robotlar.png") # assets/inputs/images/robotlar.png
    # Display the image
    #image = Resize(image, 30,interpolation=0)
    #image = Crop(image, (0,0),(500,500))
    image = Rotate(image, 15)
    

    cv2.imshow("Image", image)
    # Wait for the user to press a key
    cv2.waitKey(2000)
    # Close all windows
    cv2.destroyAllWindows()
    