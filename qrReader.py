# -*- coding: utf-8 -*-
"""
QR-Image-Sort
Image Sorting and Classification using QR codes and Google Vision API
Created by Nond and Josh on December 2, 2019
***UNDER DEVELOPMENT***
Note: This feature is under development and NOT ready for production
"""
'''
Read QR Code
Copied from https://www.learnopencv.com/opencv-qr-code-scanner-c-and-python/
'''

import cv2
import numpy as np
import sys
import time

inputImage = cv2.imread('./input/qrcode6.jpg')

# Display barcode and QR code location
def display(im, bbox):
    n = len(bbox)
    for j in range(n):
        cv2.line(im, tuple(bbox[j][0]), tuple(bbox[ (j+1) % n][0]), (255,0,0), 3)
 
    # Display results
    cv2.imshow("Results", im)
    
qrDecoder = cv2.QRCodeDetector()
 
# Detect and decode the qrcode
data,bbox,rectifiedImage = qrDecoder.detectAndDecode(inputImage)
if len(data)>0:
    print("Decoded Data : {}".format(data))
    display(inputImage, bbox)
    rectifiedImage = np.uint8(rectifiedImage);
    #cv2.imshow("Rectified QRCode", rectifiedImage);
else:
    print("QR Code not detected")
    cv2.imshow("Results", inputImage)
 
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
Text Detection using Google Cloud Vision
Packages:
    conda install -c conda-forge google-cloud-vision
    Example guide: https://cloud.google.com/vision/docs/quickstart-client-libraries
'''

def detect_text(path):
    """Detects text in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')

    for text in texts:
        print('\n"{}"'.format(text.description))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])

        print('bounds: {}'.format(','.join(vertices)))