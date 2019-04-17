# -*- coding: utf-8 -*-
"""
OCR-Image-Sort
Image Sorting and Classification via Text Detection and Recognition
Created by Nond on April 13, 2019
"""
import cv2
import numpy as np
import pytesseract
import os
import time
from imutils.object_detection import non_max_suppression

class ImageSort:
    def __init__(self):
        self.folderPath='./unsorted'
        self.sortedPath='./sorted'
        self.eastPath='./frozen_east_text_detection.pb'
        self.net=cv2.dnn.readNet(self.eastPath)
        
    def readImage(self, imgPath):        
        #read an image with OpenCV
        #this function is based on references [1,4]
        img=cv2.imread(imgPath) #default color image
        
        #config parameters for pytesseract
        #'-l eng' sets English as the language
        #'--oem 1' sets LSTM deep learning OCR Engine
        #'--psm 3' default PSM, fully automatic
        config=('-l eng --oem 1 --psm 3')
        
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
            elif len(imgToFolder)>0:
                imgToFolder.append(imgToFolder[i-1])
            else:
                print("No Key Images")
                return
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
    
    def decode_predictions(self,scores,geometry):
        # this function is copied directly from reference [3] 
        # with minor modifications
        #---(start of function from reference [3])-------------------
        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
    	# confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []
    
    	# loop over the number of rows
        for y in range(0, numRows):
    		# extract the scores (probabilities), followed by the
    		# geometrical data used to derive potential bounding box
    		# coordinates that surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]
    
    		# loop over the number of columns
            min_confidence=0.5
            for x in range(0, numCols):
    			# if our score does not have sufficient probability,
    			# ignore it
                if scoresData[x] < min_confidence:
                    continue
    			# compute the offset factor as our resulting feature
    			# maps will be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)
    
    			# extract the rotation angle for the prediction and
    			# then compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
    
    			# use the geometry volume to derive the width and height
    			# of the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]
    
    			# compute both the starting and ending (x, y)-coordinates
    			# for the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
    
    			# add the bounding box coordinates and probability score
    			# to our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])
    
    	# return a tuple of the bounding boxes and associated confidences
        return (rects, confidences)
        #---(end of function from reference [3])-------------------
        
    def textDetection(self, imgPath):
        #this function is based on references [2,3,5]
        img=cv2.imread(imgPath) #default color image
        origImg=img.copy()
        #resize image for EAST model. 
        #EAST requires width and height to be multiple of 32
        (origH, origW)=img.shape[:2]
        (newW, newH)=(320, 320)
        #ratio of scaled image to original image
        rW=origW/float(newW)
        rH=origH/float(newH)
        img=cv2.resize(img, (newW, newH))
        (H, W)=img.shape[:2]
        outputLayers=['feature_fusion/Conv_7/Sigmoid','feature_fusion/concat_3']
        #Note: EAST model is loaded when this class initiatlizes
        blob=cv2.dnn.blobFromImage(img,1.0,(W,H),(123.68, 116.78, 103.94),True,False)
        self.net.setInput(blob)
        (scores, geometry)=self.net.forward(outputLayers)
        (rects, confidences)=self.decode_predictions(scores,geometry)
        boxes=non_max_suppression(np.array(rects),probs=confidences)
        croppedImages=[]
        for (startX, startY, endX, endY) in boxes:
            #add padding to the bounding boxes, so we don't clip letters/nums
            padding=0.2
            padX=int((endX-startX)*padding)
            padY=int((endY-startY)*padding)
            
            startX=max(int(startX*rW)-padX,0)
            startY=max(int(startY*rH)-padY,0)
            endX=min(int(endX*rW)+padX,origW)
            endY=min(int(endY*rH)+padY,origH)
            
            cv2.rectangle(origImg,(startX,startY),(endX,endY),(0,255,0),8)
            croppedTextBox=origImg[startY:endY,startX:endX]
            croppedImages.append(croppedTextBox)
            #cv2.imshow('cropped',croppedTextBox)
        
        #Uncomment the 4 commands below to show the image with text detection boxes
        cv2.namedWindow('Text Recognition',cv2.WINDOW_NORMAL)
        cv2.imshow('Text Recognition',origImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        #print(boxes)
        return croppedImages
    
    def textDetectAndRecogAll(self):
        #reads all image text in the unsorted folder
        fileNames=self.getFileNames()
        imgText=[]
        for i in range(len(fileNames)):
            imgPath=self.folderPath+'/'+fileNames[i]
            croppedImages=self.textDetection(imgPath)
            
            #If there's too many text regions, this might be a
            #non-key photo with mulitple text regions
            if len(croppedImages)>4:
                continue
            
            croppedText=[]
            for j in range(len(croppedImages)):
                #config parameters for pytesseract
                #'-l eng' sets English as the language
                #'--oem 1' sets LSTM deep learning OCR Engine
                #'--psm 7' treat image as a single line of text
                config=('-l eng --oem 1 --psm 7')
                #Run Tesseract OCR
                text=pytesseract.image_to_string(croppedImages[j], config=config)
                croppedText.append(text)
            #pick the longest string found in the key image
            if len(croppedText)>0:
                imgText.append(max(croppedText,key=len))
            else:
                imgText.append('')
        #print(imgText)
        return imgText
    
    def runTextRecogOnly(self):
        start=time.time()
        fileNames=self.getFileNames()
        imgText=self.readAllUnsorted()
        self.makeFolders(imgText)
        imgToFolder=self.folderMap(imgText)
        self.sortImages(imgToFolder,fileNames)
        end=time.time()
        print(imgText)
        print('Execution time (s): ' + str(end-start))
        return
    
    def runTextDetectAndRecog(self):
        start=time.time()
        fileNames=self.getFileNames()
        imgText=self.textDetectAndRecogAll()
        self.makeFolders(imgText)
        imgToFolder=self.folderMap(imgText)
        self.sortImages(imgToFolder,fileNames)
        end=time.time()
        print(imgText)
        print('Execution time (s): ' + str(end-start))
        return
    
#test script
testImagePath1='./unsorted/IMG_2141.JPG'
testImagePath2='./unsorted/IMG_8412.JPG'
iSort=ImageSort()
#iSort.runTextRecogOnly()
#iSort.runTextDetectAndRecog()
#iSort.unsortImages(imgToFolder,fileNames)
'''
imText=iSort.readImage(testImagePath2)
print(imText)
'''