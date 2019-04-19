# OCR-Image-Sort

## Abstract

## Progress
- [x] Load and read text off of one image using tesseract.
- [x] Loop through all images in a directory and identify key photos on simple data.
- [x] Create folders based on key photo text and move photos.
- [x] Celebrate by watching Game of Thrones if everything works on simple data.
- [x] Text recognition using OpenCV.
- [x] Image cropping and passing to Tesseract.
- [x] Tesseract error handling.
- [ ] Evaluate algo on medium sized dataset and adjust parameters
- [ ] Evaluate algorithm on full complex image dataset.
- [ ] Refine key photo identification and key photo likelihood.
- [ ] Build a GUI interface.
- [ ] Speed up text recognition using TensorFlow.

## OCR-Image-Sort Overview
The proposed system will process a sequential set of equipment images and identify key photos. 
A key photo is a photo that contains the tag or nameplate of an equipment and is the first photo in a photo set of that equipment. 
In the above example, key photos are p_1^1,p_1^2,and p_1^3. 
The system will be built to run on a personal desktop or laptop computer, but not on a mobile platform such as a phone or camera. 
The system will label all the photos associated with the key photo with the appropriate tag and sequence up to the next key photo (see figure 1 below).

The system will be based on three main technologies: Python3, OpenCV [1], and Tesseract [2]. 
Python3 will be the general wrapper for all system components and will sort and label the photos. 
OpenCV will be used to identify areas in each photo that contain text. 
Tesseract will be used to transcribe the text in each area identified by OpenCV [3]. 
Python3 will evaluate the likelihood that the photo is a key photo. 

## Notes

### Package Installation
1. Install tesseract for windows from https://github.com/UB-Mannheim/tesseract/wiki
2. Add tesseract to PATH variable for windows (https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10/)
3. Install latest version of anaconda
4. Open Anaconda Prompt and install the following packages
```
conda install pip
pip install tesseract
pip install pytesseract
pip install opencv-contrib-python
pip install imutils
conda install tensorflow-gpu
```

### Main APIs and Software Packages:
1. https://github.com/opencv/opencv
2. https://github.com/tesseract-ocr/tesseract
3. https://github.com/madmaze/pytesseract
4. Python 3.6.2 |Anaconda custom (64-bit)| via https://www.anaconda.com/
5. Spyder IDE: https://www.spyder-ide.org/
6. EAST Text Detection model: https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1
7. https://github.com/argman/EAST

### References:
1. V. Singh Chandel and S. Mallick, "Deep Learning based Text Recognition (OCR) using Tesseract and OpenCV," ed. https://www.learnopencv.com/deep-learning-based-text-recognition-ocr-using-tesseract-and-opencv/
2. V. Shrimali and S. Mallick, "Deep Learning based Text Detection Using OpenCV," ed. https://www.learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/
3. https://www.pyimagesearch.com/2018/09/17/opencv-ocr-and-text-recognition-with-tesseract/
4. R. Smith, "An Overview of the Tesseract OCR Engine," presented at the Proceedings of the Ninth International Conference on Document Analysis and Recognition - Volume 02, 2007.
5. X. Zhou et al., "EAST: an efficient and accurate scene text detector," in Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, 2017, pp. 5551-5560.

### Other threads consulted:
1. The OpenCV Library. (2000). Accessed: February 28, 2019. [Online]. Available: https://opencv.org/
2. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," Journal of machine learning research, vol. 12, no. Oct, pp. 2825-2830, 2011.
3. M. Abadi et al., "TensorFlow: Large-scale machine learning on heterogeneous systems," ed, 2015.
4. Caffe: Convolutional Architecture for Fast Feature Embedding. (2014). [Online]. Available: http://caffe.berkeleyvision.org/
5. A. Paszke et al., "Automatic differentiation in pytorch," 2017.
6. https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html
7. https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory-in-python
8. https://github.com/argman/EAST
9. https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/