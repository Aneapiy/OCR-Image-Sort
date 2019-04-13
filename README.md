# OCR-Image-Sort

## Abstract

## Progress

## OCR-Image-Sort Overview
The proposed system will process a sequential set of equipment images and identify key photos. A key photo is a photo that contains the tag or nameplate of an equipment and is the first photo in a photo set of that equipment. In the above example, key photos are p_1^1,p_1^2,and p_1^3. The system will be built to run on a personal desktop or laptop computer, but not on a mobile platform such as a phone or camera. The system will label all the photos associated with the key photo with the appropriate tag and sequence up to the next key photo (see figure 1 below).
The system will be based on three main technologies: Python3, OpenCV [1], and Tesseract [2]. Python3 will be the general wrapper for all system components and will sort and label the photos. OpenCV will be used to identify areas in each photo that contain text. Tesseract will be used to transcribe the text in each area identified by OpenCV [3]. Python3 will evaluate the likelihood that the photo is a key photo. 

## Notes

### Main APIs:
1. https://github.com/opencv/opencv
2. https://github.com/tesseract-ocr/tesseract
3. https://github.com/madmaze/pytesseract

### References:
1. V. Singh Chandel and S. Mallick, "Deep Learning based Text Recognition (OCR) using Tesseract and OpenCV," ed. https://www.learnopencv.com/deep-learning-based-text-recognition-ocr-using-tesseract-and-opencv/
2. V. Singh Chandel and S. Mallick, "Deep Learning based Text Detection Using OpenCV," ed. https://www.learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/
3. R. Smith, "An Overview of the Tesseract OCR Engine," presented at the Proceedings of the Ninth International Conference on Document Analysis and Recognition - Volume 02, 2007.

### Other threads consulted:
1. The OpenCV Library. (2000). Accessed: February 28, 2019. [Online]. Available: https://opencv.org/
2. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," Journal of machine learning research, vol. 12, no. Oct, pp. 2825-2830, 2011.
3. M. Abadi et al., "TensorFlow: Large-scale machine learning on heterogeneous systems," ed, 2015.
4. Caffe: Convolutional Architecture for Fast Feature Embedding. (2014). [Online]. Available: http://caffe.berkeleyvision.org/
5. A. Paszke et al., "Automatic differentiation in pytorch," 2017.