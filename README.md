# DocumentScanner
Image-based document scanner built using OpenCV-Python


# Summary 
The program requests that the user input the file path to the image of the document to be scanned. Provided that the document is the primary focal point of the image, has a rectangular shape with four pronounced corners, and reasonably contrasts with the background, the program should be able create a PDF of the document from the image. This document is then named by the user and saved in the user's current working directory. The program also displays the document contour drawn over the original image, as well as an image of the final transformation. These images are displayed until the user presses a keyboard key, at which point the images are automatically closed. This is done to show any possible error the program may have had while finding the contours. Possible errors may result from the document not having proper contrast with its background, and the user may need to try a different image. 
