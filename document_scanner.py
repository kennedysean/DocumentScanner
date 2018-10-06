import numpy as np
import cv2 as cv
from PIL import Image
from fpdf import FPDF

"""  
Image-Based Document Scanner Using OpenCV-Python  
Sean Kennedy
https://github.com/kennedysean/DocumentScanner
"""

def resize(image, size):
    """Resizes the image to the desired size"""
    
    original_height = image.shape[0]
    original_width = image.shape[1]
    ratio = original_width/original_height
    resized_image = cv.resize(image,(int(size*ratio),size))
    
    return resized_image

def get_contour(image):
    """Finds the edges of the document, returns the coordinates of the 
    document contour, and returns the original image with the contour lines 
    drawn over the image"""
    
    # make a copy of the original image to be used later in the function
    original_image = image.copy()
    # apply grayscale, blur the image, and find the edges
    image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    image = cv.GaussianBlur(image,(5,5),0)
    image = cv.Canny(image,75,200)
    # find all contours in image
    contours_image, contours, hierarchy = cv.findContours(image,cv.RETR_LIST, \
                                                    cv.CHAIN_APPROX_SIMPLE)
    # sort contours from largest to smallest
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    # loop through contours to find document contour
    for contour in contours:
        # approximate the contour
        edge = cv.arcLength(contour, True)
        edge_points = cv.approxPolyDP(contour, 0.025 * edge, True)
     
        # if the contour has four points, the document is assumed to be found
        if len(edge_points) == 4:
            doc_contour = edge_points
            print("\nRectangular contour detected.")
            break
    
    if len(edge_points) != 4:
        return 0, 0
    
    # draw the document contour over the original image
    cv.drawContours(original_image, [doc_contour], -1, (255, 0, 0), 3)
    
    # return the coordinates of the document contour and the original image 
    # with the contour lines drawn over top of the image
    return doc_contour, original_image

def rect_transform(image, pts):
    """Transforms the image to obtain a top-down view of the document"""
    
    # assign rectangular coordinates to their specific corners
    s = pts.sum(axis = 1)
    diff = np.diff(pts, axis = 1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    # place corner coordinates in a sorted array
    rect = np.array([tl, tr, br, bl], dtype= "float32")
    # compute the width of the new image
    bottom_width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    top_width = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width = max(int(bottom_width), int(top_width))
    # compute the height of the new image
    right_height = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    left_height = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    height = max(int(right_height), int(left_height))
    # construct destination points for transformed image
    target = np.array([[0, 0],[width - 1, 0],[width - 1, height - 1],
                    [0, height - 1]], dtype = "float32")
    
    # apply the transformation and return the image
    M = cv.getPerspectiveTransform(rect, target)
    transformed_image = cv.warpPerspective(image, M, (width, height))
    return transformed_image

def kernel_sharpen(image):
    """Creates and applies sharpening kernel filter to image"""
    
    # create sharpening kernel
    kernel = np.ones( (9,9), np.float32) / -90.0
    kernel[4,4] = 2.0
    # apply kernel filter to the image parameter
    filtered_image = cv.filter2D(image, -1, kernel)
    
    # return sharpened image
    return filtered_image

def adaptive_threshold(adjusted_image):
    """Applies adaptive threshold to image (not used in main function)"""
    
    # convert image to grayscale
    grayscale = cv.cvtColor(adjusted_image, cv.COLOR_BGR2GRAY)
    # apply adaptive threshold
    adap = cv.adaptiveThreshold(grayscale,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,5)
    
    #return adap
    return adap

def jpg_to_pdf(image_name):
    """Converts a JPG file into a PDF file saved in the current directory
    NOTE: image_name string should not include .jpg file extension"""
    
    # create and format PDF to fit image
    image = Image.open(image_name+".jpg")
    width, height = image.size
    pdf = FPDF(unit = "pt", format = [width, height])
    # add image to pdf and save
    pdf.add_page()
    pdf.image(image_name+".jpg", 0, 0)
    pdf.output(image_name+".pdf", "F")

def main():

    # find the image of the document to be scanned
    image_file = input("Enter file path of image (JPG) to be scanned: ")
    image = cv.imread(image_file)
    
    # resize original image to appropriate size for finding document contour
    image = resize(image, 1000)
    
    # find and draw contour for the document.
    doc_contour, contour_image = get_contour(image)
    
    # check that a rectangular contour was found
    try:
        if doc_contour==0:
            print("\nRectangular contour could not be detected. \
                  Program terminated.")
    except ValueError:
        scanned_document = input("Enter name of new scanned document (Do not \
                                include any file extensions in the name): ")
        
        # show original image with contour drawn on edges of document
        cv.imshow("Document Edges",contour_image)
        
        # apply perspective transformation and show the resulting image
        transformed = rect_transform(image, doc_contour.reshape(4, 2))
        
        # resize the image of the document
        resized_image = resize(transformed, 842)
        
        # apply grayscale and kernel sharpening to image and show the result
        final_image = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
        final_image = kernel_sharpen(final_image)
        cv.imshow("Final Document Image",final_image)
        
        # save the JPG image in the current directory
        cv.imwrite(scanned_document+".jpg",final_image)
        
        # convert the final image into a PDF
        jpg_to_pdf(scanned_document)
        
        # display images until a key is pressed.
        cv.waitKey(0)
        cv.destroyAllWindows()
        
if __name__ == "__main__":
    main()