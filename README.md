# DocumentScanner
Image-based document scanner program built using OpenCV-Python
  
  
## Summary 
The program requests that the user input the file path to the image of the document to be scanned. Provided that the document is the primary focal point of the image, has a rectangular shape with four pronounced corners, and reasonably contrasts with the background, the program should be able create a PDF of the document from the image. This document is then named by the user and saved in the user's current working directory. The program also displays the document contour drawn over the original image, as well as an image of the final transformation. These images are displayed until the user presses a keyboard key, at which point the images are automatically closed. This is done to show any possible error the program may have had while finding the document edges. Possible errors may result from the document not having proper contrast with its background, and the user may need to try a different image.  
  
  
## Algorithm
The program must first detect all of the edges in the image. This is done using the Canny Edge Detection algorithm provided by OpenCV after first converting the image to grayscale and applying a Gaussian filter to reduce noise. The program then uses the edged image to detect the edges of the document. This is done by using OpenCV to find all of the contours in the edged image and finding the largest contour such that the contour is defined by four edges. The program safely assumes that the document is rectangular and is the main focus of the image. Once the document contour has been found, a transformation is applied to the image to obtain a cropped, top-down view of the document. This is done by identifying the specific corners of the document contour, finding the target corners of the document using the calculated height and width, and using OpenCV Perspective Transform to calculate and apply the tranformation matrix to the image. Lastly, a kernel sharpening filter is applied to give the image a sharper appearance, and the final image is converted to PDF format.  
  
NOTE: I had originally used adaptive thresholding in the program to give the PDF a sharper contrast. However, I found that a static thresholding filter wasn't consistent across different image qualities, so I decided to use kernel sharpening instead to provide a similar effect over a wider variety of images. I still leave the adaptive thresholding function in the source code, but do not use it in the main program.  
  
  
## Python Libraries
* OpenCV (Open Source Computer Vision Library)
* NumPy
* Python Imaging Library (PIL)
* PyFPDF
