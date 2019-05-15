# OCR-Image-Sort
Image Sorting and Classification via Clustering, Text Detection, and Text Recognition

## Abstract
In this paper, I propose a system for automatically sorting images into folders by using text detection and text recognition to extract descriptors from key images. The input dataset must be a sequence of images where an image containing the text descriptor for a subset of images precedes the subset in the overall sequence. The system uses k-Means clustering on grayscale image histogram data to filter potential key images from the dataset. Potential key images are then passed into the EAST deep learning CNN model and OpenCV to detect text regions in key images. These text regions then go through Tesseract’s LSTM deep learning engine for text recognition. Afterwards, Python wrappers use the output from Tesseract to create folders on disk and move images into the appropriate folders. The OCR Image Sort system has been evaluated on a 3.9 GB dataset of 1433 images taken in North American refineries resulting in a precision of 83.1% and recall of 90.2%. Of the correct true positive key images, 81.1% had perfect text recognition and 18.9% had text recognition errors.

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
The proposed system processes a sequential set of equipment images and identifies key images. A key image is a photo that contains the tag or nameplate of an equipment and is the first photo in a photo set of that equipment. In the above example, key photos are p_1^1,p_1^2,and p_1^3. The system is built to run on a personal desktop or laptop computer, but not on a mobile platform such as a phone or camera. The system will sort all the photos associated with the key photo into a single folder.

The system is based on six main technologies: Python 3 [9], Tesseract [4], the EAST model [5], OpenCV [6], PyTesseract[7], and scikit-Learn [8]. Python3 is the general wrapper for all system components and interacts with the operating system to create folders and move image files around. Image filtering to identify key images is done via scikit-learn’s k-Means clustering algorithm. Text detection is done by OpenCV with the EAST model. After OpenCV identifies the text regions, Tesseract does the text recognition and outputs a string back to Python3. PyTesseract provides a Python wrapper for Tesseract.

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
### Running the program from a Terminal:
Create a folder called ./input/ and another folder called ./output/ in the same directory as the code. Make sure the folder paths in the ImageSort class initialization for the input and output folders point to the correct location. Check that the location of the EAST model is correct.
To run the program from the terminal, make sure that the terminal directory as the folder where imagesort.py, the EAST model, and the input and output folders are located. Run the program using:

> python imagesort.py

No additional inputs or options are required.

### Running the program from a Python 3 console for OCR only mode:
First, comment out the default run script at the bottom of the imagesort.py file. Run imagesort.py to load in the code. Initialize the class with.
> iSort=ImageSort()
To run the code with text detection + OCR (recommended), run the following command in the Python terminal after initializing the class:
> iSort.runDefault()

To run the code with OCR only (not recommended), run the following command in the Python terminal after initializing the class:

> iSort.runTextRecogOnly()

## Notes

### Code Overview
The software is contained in one class called ImageSort. The location of the EAST model, the unsorted photos folder path, and the sorted photos folder path is initialized with the class. The EAST model is also loaded into memory during class initialization.
The ImageSort class has two main execution modes: text detection with OCR mode and an OCR only mode. The text detection with OCR mode is the default and recommended execution mode as this mode is faster and produces more accurate results. This mode uses OpenCV to extract a greyscale image histogram of each image, scikit-learn’s k-Means clustering algorithm to identify potential key images based on the histogram of each image, OpenCV and the EAST model to detect text regions in images, and then passes the text regions to Tesseract for recognition. The OCR only mode passes the entirety of all images to Tesseract. This mode is slower as tesseract has to process the entire image instead of just the text region and may also produce more false positive key image identifications.

#### Text Detection with OCR mode:
The main function for the text detection with OCR mode is runDefault().
This function first starts the execution timer then calls getFileNames() which scans the folder ./input/ and returns a list of all the image filenames in that folder. Next, the function calls getHisto() which uses OpenCV to produce a an array containing greyscale image histograms of all images (see Fig. 3). The histogram contains 20 bins with each pixel value ranging from 0 to 256. These parameters were determined experimentally. The array of histograms is passed into the findKeyPhotos() function which fits a k-Means clustering algorithm with 2 clusters to the histogram data. Each bin in the histogram data is considered a feature in the k-Means clustering algorithm. The function returns the cluster assignments of each image.
Under the assumption that there are less key images than non-key images in the dataset, the function textDetectAndRecogAll() loops through all image files in the smaller cluster and calls the textDetection function. The textDetection function uses the EAST model to identify regions of text within the image and then passes these text regions into Tesseract for text recognition.
The text output from Tesseract gets appended to a list of all text strings within that image and the longest string of text is returned as the most likely label for that potential key image. This list of key image text is then used by the makeFolders function to create folders in the ./output/ folder named after each key image text. The folderMap function takes the key image text list and maps where each image in the ./intput/ folder should go in the ./output/ folder based on which key image precedes each regular image. The sortImages function uses this map to move all of the images from the ./input/ folder into the appropriate subfolder in ./output/ 

#### OCR only mode:
The main function for the OCR only mode is runTextRecogOnly(). This function is similar to the main function for the text detection with OCR mode except for the function creates the list of key image text using readAllUnsorted() instead of textDetectandRecogAll(). In readAllUnsorted(), the function loops through all the images and calls Tesseract on the full image instead of on small cropped images passed from a text detector. After the key image text list is created, the program runs through the same functions as above to create folders and move image files.

### Preliminary Evaluation
I used Python’s native time.time() method to test the execution time. Note that the time it takes to load the EAST model into memory is not included in these measurements since the model is loaded into memory when the class object is created and not reloaded each time the script is executed.

For the full system evaluation, I used the OCR Image Sort system to sort a 3.9 GB dataset of 1433 images of over 100 different pieces of equipment (sensors, control valves, piping, enclosures, etc) taken in North American refineries and compared the system’s automatic sort to a manual sort conducted by a subject matter expert. Each image has a resolution of 4608 by 3456 pixels, is stored in JPEG format, and ranges from 1.5MB to 4.3MB in size. The system sorted 1433 images in 327.984 seconds (0.229 seconds per image). The results are in Table 1 below.

The system yielded a precision of 83.1% with a recall of 90.2%. Of the 74 correct key image identifications, 60 images (81.1%) had perfect text recognition and 14 images (18.9%) had text recognition errors. These results are a significant improvement over the prototype system. The prototype system had a precision of < 50% and a recall of ~75%. The prototype also took 0.478 seconds per image to process images at a lower resolution.

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
This system could also be used for other applications outside of refinery image sorting that involve segregating a sequence of images by key images in the sequence. School yearbook photographers that take photos of students where the first photo of each student is the student holding a name card could use this system to automatically sort and label photos from the shoot. Car dealerships could use this system to automatically sort photos of taken of incoming car inventory. Travelers could use this system to sort photos of cities, towns, and tourist attractions that they visit based on photos of different signage.

### Main APIs and Software Packages:
1. https://github.com/opencv/opencv
2. https://github.com/tesseract-ocr/tesseract
3. https://github.com/madmaze/pytesseract
4. Python 3.7.3 |Anaconda custom (64-bit)| via https://www.anaconda.com/
5. Spyder IDE: https://www.spyder-ide.org/
6. EAST Text Detection model: https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1
7. https://github.com/argman/EAST
8. https://github.com/UB-Mannheim/tesseract/wiki
9. https://scikit-learn.org/stable/modules/clustering.html#k-means

### References:
1. V. Singh Chandel and S. Mallick, "Deep Learning based Text Recognition (OCR) using Tesseract and OpenCV," Learn OpenCV, 2018. Available: https://www.learnopencv.com/deep-learning-based-text-recognition-ocr-using-tesseract-and-opencv/
2. V. Shrimali and S. Mallick, "Deep Learning based Text Detection Using OpenCV," Learn OpenCV, 2019. Available: https://www.learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/
3. A. Rosebrock, “OpenCV OCR and text recognition with Tesseract,” pyimagesearch, 2019. Available: https://www.pyimagesearch.com/2018/09/17/opencv-ocr-and-text-recognition-with-tesseract/
4. R. Smith, "An Overview of the Tesseract OCR Engine," presented at the Proceedings of the Ninth International Conference on Document Analysis and Recognition - Volume 02, 2007. Available: https://github.com/tesseract-ocr/tesseract
5. X. Zhou et al., "EAST: an efficient and accurate scene text detector," in Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, 2017, pp. 5551-5560. Available: https://github.com/argman/EAST
6. The OpenCV Library. (2000). Accessed: February 28, 2019. [Online]. Available: https://opencv.org/
7. M. Lee et al., “A Python Wrapper for Google Tesseract,” Python Tesseract. (2019). Accessed: May 13, 2019. [Online]. Available: https://github.com/madmaze/pytesseract
8. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," Journal of machine learning research, vol. 12, no. Oct, pp. 2825-2830, 2011.
9. Python Software Foundation. Python Programming Language, version 3.7.3. Available: https://www.python.org/


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