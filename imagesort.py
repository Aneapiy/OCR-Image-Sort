# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 13:24:43 2019

@author: nhasbamr
"""
import cv2
import sys
import pytesseract

class ImageSort:
    def readImage(self, imPath):        
        #read an image with OpenCV
        img=cv2.imread(imPath) #default color image
        
        #config parameters for pytesseract
        #'-l eng' sets English as the language
        #'--oem 1' sets LSTM deep learning OCR Engine
        config=('-l eng --oem 1')
        
        #Run Tesseract OCR
        text=pytesseract.image_to_string(img, config=config)
        
        return text


#test script
testImagePath1='./testImages/image1.png'
testImagePath2='./testImages/image2.jpg'
iSort=ImageSort()
imText=iSort.readImage(testImagePath2)
print(imText)