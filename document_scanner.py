import numpy as np
import cv2 as cv
from PIL import Image
from fpdf import FPDF

"""  Document Scanner Using OpenCV-Python  """


def resize(image):
    """Resizes the image so that it can be properly displayed"""
    
    original_height=image.shape[0]
    original_width=image.shape[1]
    ratio=original_width/original_height
    resized_image=cv.resize(image,(int(1000*ratio),1000))
    
    return resized_image

def get_contour(image):
    """Finds the edges of the document, returns the coordinates of the 
    document contour, and returns the original image with the contour lines 
    drawn over the image"""
    
    # make a copy of the original image to be used later in the function
    original_image=image.copy()
    # apply grayscale, blur the image, and find the edges
    image=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    image=cv.GaussianBlur(image,(5,5),0)
    image=cv.Canny(image,75,200)
    # find all contours in image
    contours_image, contours, hierarchy = cv.findContours(image,cv.RETR_LIST, \
                                                    cv.CHAIN_APPROX_SIMPLE)
    # sort contours from largest to smallest
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    # loop through contours to find document contour
    for contour in contours:
        # approximate the contour
        edge = cv.arcLength(contour, True)
        edge_points = cv.approxPolyDP(contour, 0.02 * edge, True)
     
        # if the contour has four points, the document is assumed to be found
        if len(edge_points) == 4:
            doc_contour = edge_points
            print("Document edges found.")
            break
    # draw the document contour over the original image
    cv.drawContours(original_image, [doc_contour], -1, (255, 0, 0), 5)
    
    # return the coordinates of the document contour and the original image 
    # with the contour lines drawn over top of the image
    return doc_contour, original_image

def rect_transform(image, pts):
    """Transforms the image to obtain a top-down view of the document"""
    # https://www.pyimagesearch.com/2014/08/25/4-
    # point-opencv-getperspective-transform-example/
    
    # sum coordinates of each point to find top-left and bottom-right corners
    s = pts.sum(axis = 1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
	 # subtract coordinates for each point to find top-right and bottom-left
    diff = np.diff(pts, axis = 1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    # place corner coordinates in an array
    rect=np.array([tl, tr, br, bl], dtype= "float32")
    # compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # construct destination points for transformed image
    dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype = "float32")
    
    # compute the perspective transform matrix, apply it, and return the image
    M = cv.getPerspectiveTransform(rect, dst)
    transformed_image = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    return transformed_image

def adaptive_threshold(adjusted_image):
    """Applies adaptive threshold to image"""
    
    # convert image to grayscale
    grayscale = cv.cvtColor(adjusted_image, cv.COLOR_BGR2GRAY)
    # apply adaptive threshold
    adap = cv.adaptiveThreshold(grayscale,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,13,6)
    
    return adap

def jpg_to_pdf(image_name):
    """Converts a JPEG file into a PDF file saved in the current directory"""
    
    # create and format PDF to fit image
    image = Image.open(image_name+".jpg")
    width, height = image.size
    pdf = FPDF(unit = "pt", format = [width, height])
    # add image to pdf and save
    pdf.add_page()
    pdf.image(image_name+".jpg", 0, 0)
    pdf.output(image_name+".pdf", "F")

def main():

    # Find the image of the document to be scanned.
    image_file=input("Enter file path of image (JPG) to be scanned: ")
    image=cv.imread(image_file)
    
    # Resize image of document.
    resized_image=resize(image)
    
    # Find and draw contour for the document.
    doc_contour, contour_image = get_contour(resized_image)
    
    # Show original image with contour drawn on edges of document.
    cv.imshow("Contour",contour_image)
    
    # Apply perspective transformation to the image.
    transformed = rect_transform(resized_image, doc_contour.reshape(4, 2))
    cv.imshow("Transformed",transformed)
    # Apply adaptive threshold to the final image.
    final_image = adaptive_threshold(transformed)
    
    cv.imshow("final_image",final_image)
    # Save the JPG image in the current directory.
    cv.imwrite("new_document.jpg",final_image)
    
    # Convert the final image to a PDF.
    jpg_to_pdf("new_document")
    
    # Display images until a key is pressed.
    cv.waitKey(0)
    cv.destroyAllWindows()
        
if __name__ == "__main__":
    main()