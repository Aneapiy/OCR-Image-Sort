# OCR-Image-Sort
This code is in its infancy! Things will break! Good luck, have fun!

## Abstract
OCR-Image-Sort automatically sorts images into folders by using text detection and text recognition to extract descriptors from key images. 
The input dataset must be a sequence of images where an image containing the text descriptor for a subset of images precedes the subset in the overall sequence. 
The system uses the EAST deep learning CNN model and OpenCV to detect text regions in key images and passes these text regions to Tesseract’s LSTM deep learning engine for text recognition. 
Python functions then use the output from Tesseract to create folders on disk and moves the images into the appropriate folder. 

## Progress
- [x] Load and read text off of one image using tesseract.
- [x] Loop through all images in a directory and identify key photos on simple data.
- [x] Create folders based on key photo text and move photos.
- [x] Celebrate by watching Game of Thrones if everything works on simple data.
- [x] Text recognition using OpenCV.
- [x] Image cropping and passing to Tesseract.
- [x] Basic Tesseract error handling.
- [x] Evaluate algorithm on medium sized dataset and adjust parameters
- [x] Evaluate algorithm on full complex image dataset.
- [x] Refine key photo identification and key photo likelihood.
- [ ] Build a GUI interface.

## OCR-Image-Sort Overview
The proposed system process a sequential set of images and identify key photos. 
A key photo is a photo that contains the tag or nameplate of an equipment and is the first photo in a photo set of that equipment. 
The system is built to run on a personal desktop or laptop computer, but not on a mobile platform such as a phone or camera. 
The system will sort all the photos associated with the key photo into a single folder (see figure 1 below).

The system is based on four main technologies: Python3, OpenCV [6], the EAST model [5], and Tesseract [4]. 
Python3 is the general wrapper for all system components and interacts with the operating system to create folders and move image files around. 
Text detection is done by OpenCV with the EAST model. 
After OpenCV identifies the text regions, Tesseract does the text recognition and outputs a string back to Python3. 

## Package Installation
1. Install tesseract for windows from https://github.com/UB-Mannheim/tesseract/wiki
2. Add tesseract to PATH variable for windows (https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10/)
3. Install the latest version of Anaconda
4. Open Anaconda Prompt and install the following packages
```
conda install pip
pip install tesseract
pip install pytesseract
pip install opencv-contrib-python
pip install imutils
conda install tensorflow-gpu
```

## Code Execution Instructions

Make sure the folder paths for the unsorted and sorted folders point to the correct location. 
Put all images in the ./unsorted/ folder.
Check that the location of the EAST model is correct.
Initialize the class with
> iSort=ImageSort()

To run the code with text detection + OCR (recommended), run the following command in the Python terminal after initializing the class:
> iSort.runDefault()

To run the code with OCR only (not recommended), run the following command in the Python terminal after initializing the class:
> iSort.runTextRecogOnly()

All the images should end up in the ./sorted/ folder under subfolders named after descriptors from key images.

## Notes

### Code Overview
Everything is contained in one class called ImageSort. 
The location of the EAST model, the unsorted photos folder path, and the sorted photos folder path is initialized with the class. 
The EAST model is also loaded into memory during class initialization.

#### Text Detection with OCR mode:
The main function for the text detection with OCR mode is runTextDetectAndRecog(). 
This function first starts the execution timer then calls getFileNames() which scans the ./unsorted/ folder and returns a list of all the image filenames in that folder. 
The list of filenames then gets passed to textDetectAndRecogAll() which loops through each file and calls the textDetection function. The textDetection function uses the EAST model to identify regions of text within the image and then passes these text regions into Tesseract for text recognition. 
The text output from Tesseract gets appended to a list of all text within that image and the longest string of text is returned as the most likely label for that possible key image. This list of key image text is then used by the makeFolders function to create folders in the /sorted/ folder named after each key image text. 
The folderMap function takes the key image text list and maps where each image in the /unsorted/ folder should go based on which key image precedes each regular image. 
The sortImages function uses this map to move all of the images from the /unsorted/ folder into the appropriate subfolder in /sorted/.

#### OCR only mode:
The main function for the OCR only mode is runTextRecogOnly(). 
This function is similar to the main function for the text detection with OCR mode except for the function creates the list of key image text using readAllUnsorted() instead of textDetectandRecogAll(). 
In readAllUnsorted(), the function loops through all the images and calls Tesseract on the full image instead of on small text detection regions. 
After the key image text list is created, the program runs through the same functions as above to create folders and move image files.

### Preliminary Evaluation
I used Python’s native time.time() method to test the execution time of both run modes. Note that the time it takes to load the EAST model into memory is not included in these measurements since the model is loaded into memory when the class object is created and not reloaded each time the script is executed. The small scale test with 6 images (2 key images and 4 regular images) resulted in the following execution times:
- OCR only execution mode: 17.49 sec  = 2.915 sec/image
- Text detection + OCR execution mode: 2.87 sec = 0.478 sec/image

The prototype was tested on a personal computer with the following specifications:
- OS: Windows 7 Service Pack 1 64-bit 
- CPU: Intel Core i5-4690K 3.50 GHz Quad Core
- GPU: Nvidia GeForce GTX 1080
- Ram: 16.0 GB
- OS Drive: Samsung SSD 850 EVO
- Program Drive: Toshiba SSD OCZ Trion 150 
- Motherboard: MSI Z97 Gaming 5

During the test, the computer was air cooled and not overclocked. The ambient temperature of the testing room was approximately 70-72°F.

### Possible Applications
This system could be used for other applications that involve segregating a sequence of images by key images in the sequence. 
School yearbook photographers that take photos of students where the first photo of each student is the student holding a name card could use this system to automatically sort and label photos from the shoot.
Car dealerships could use this system to automatically sort photos of taken of incoming car inventory.
Travelers could use this system to sort photos of cities, towns, and tourist attractions that they visit based on photos of different signage.

### Main APIs and Software Packages:
1. https://github.com/opencv/opencv
2. https://github.com/tesseract-ocr/tesseract
3. https://github.com/madmaze/pytesseract
4. Python 3.6.2 |Anaconda custom (64-bit)| via https://www.anaconda.com/
5. Spyder IDE: https://www.spyder-ide.org/
6. EAST Text Detection model: https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1
7. https://github.com/argman/EAST
8. https://github.com/UB-Mannheim/tesseract/wiki
9. https://scikit-learn.org/stable/modules/clustering.html#k-means

### References:
1. V. Singh Chandel and S. Mallick, "Deep Learning based Text Recognition (OCR) using Tesseract and OpenCV," ed. https://www.learnopencv.com/deep-learning-based-text-recognition-ocr-using-tesseract-and-opencv/
2. V. Shrimali and S. Mallick, "Deep Learning based Text Detection Using OpenCV," ed. https://www.learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/
3. https://www.pyimagesearch.com/2018/09/17/opencv-ocr-and-text-recognition-with-tesseract/
4. R. Smith, "An Overview of the Tesseract OCR Engine," presented at the Proceedings of the Ninth International Conference on Document Analysis and Recognition - Volume 02, 2007. https://github.com/tesseract-ocr/tesseract
5. X. Zhou et al., "EAST: an efficient and accurate scene text detector," in Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, 2017, pp. 5551-5560. https://github.com/argman/EAST
6. The OpenCV Library. (2000). Accessed: February 28, 2019. [Online]. Available: https://opencv.org/
7. https://github.com/madmaze/pytesseract
8. M. Abadi et al., "TensorFlow: Large-scale machine learning on heterogeneous systems," ed, 2015.
9. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," Journal of machine learning research, vol. 12, no. Oct, pp. 2825-2830, 2011.

### Other threads consulted:
1. The OpenCV Library. (2000). Accessed: February 28, 2019. [Online]. Available: https://opencv.org/
2. https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html#sphx-glr-auto-examples-cluster-plot-kmeans-assumptions-py
3. M. Abadi et al., "TensorFlow: Large-scale machine learning on heterogeneous systems," ed, 2015.
4. Caffe: Convolutional Architecture for Fast Feature Embedding. (2014). [Online]. Available: http://caffe.berkeleyvision.org/
5. A. Paszke et al., "Automatic differentiation in pytorch," 2017.
6. https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html
7. https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory-in-python
8. https://github.com/argman/EAST
9. https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
10. https://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html
11. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans