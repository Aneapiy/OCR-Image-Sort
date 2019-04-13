# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 13:24:43 2019

@author: nhasbamr
"""
import cv2
import sys
import pytesseract
import os
import time

class ImageSort:
    def __init__(self):
        self.folderPath='./unsorted'
        self.sortedPath='./sorted'
        
    def readImage(self, imgPath):        
        #read an image with OpenCV
        #this function is based on reference [1]
        img=cv2.imread(imgPath) #default color image
        
        #config parameters for pytesseract
        #'-l eng' sets English as the language
        #'--oem 1' sets LSTM deep learning OCR Engine
        config=('-l eng --oem 1')
        
        #Run Tesseract OCR
        text=pytesseract.image_to_string(img, config=config)
        
        return text

    def getFileNames(self):
        #creates list of image file names
        fileNames=os.listdir(self.folderPath)
        return fileNames
    
    def readAllUnsorted(self):
        #reads all image text in the unsorted folder
        fileNames=self.getFileNames()
        numOfImgs=len(fileNames)
        imgText=[]
        for i in range(numOfImgs):
            imgPath=self.folderPath+'/'+fileNames[i]
            imgText.append(self.readImage(imgPath))
        return imgText
    
    def makeFolders(self,imgText):
        #make folders based on image text
        for i in range(len(imgText)):
            if len(imgText[i])>0:
                folderName=self.sortedPath+'/'+imgText[i]
                try:
                    os.makedirs(folderName)
                except FileExistsError:
                    #directory of the same name already exists
                    pass
        return
    
    def folderMap(self,imgText):
        #creates a map of what folders which image goes to
        imgToFolder=[]
        for i in range(len(imgText)):
            if len(imgText[i])>0:
                folderName=self.sortedPath+'/'+imgText[i]
                imgToFolder.append(folderName)
            else:
                imgToFolder.append(imgToFolder[i-1])
        return imgToFolder
    
    def sortImages(self,imgToFolder,fileNames):
        #takes the folder map and moves the images to that folder
        #the folder map and the file name list must be in the same order
        for i in range(len(imgToFolder)):
            source_file=self.folderPath+'/'+fileNames[i]
            destination_file=imgToFolder[i]+'/'+fileNames[i]
            os.rename(source_file,destination_file)
        return
    
    def unsortImages(self,imgToFolder,fileNames):
        #moves all images back to the unsorted folder
        for i in range(len(imgToFolder)):
            source_file=self.folderPath+'/'+fileNames[i]
            destination_file=imgToFolder[i]+'/'+fileNames[i]
            os.rename(destination_file,source_file)
        return
        
#test script
testImagePath1='./unsorted/image1.png'
testImagePath2='./unsorted/image2.jpg'
iSort=ImageSort()
start=time.time()
fileNames=iSort.getFileNames()
imgText=iSort.readAllUnsorted()
iSort.makeFolders(imgText)
imgToFolder=iSort.folderMap(imgText)
iSort.sortImages(imgToFolder,fileNames)
#iSort.unsortImages(imgToFolder,fileNames)
end=time.time()
print(imgText)
print('Execution time (s): ' + str(end-start))
'''
imText=iSort.readImage(testImagePath2)
print(imText)
'''