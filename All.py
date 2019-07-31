import cv2
import pytesseract
import numpy as np
import re
import os, sys
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

def opencv_gray():
     image = cv2.imread("test2.png")
     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     cv2.imshow("Original", image)
     cv2.imshow("Original- gray", gray_image)
     cv2.waitKey(0)
     cv2.destroyAllWindows()


def resize():
     img = Image.open('test.png') # image extension *.png,*.jpg
     new_width  = 200
     new_height = 300
     img = img.resize((new_width, new_height), Image.ANTIALIAS)
     img.save('resized.png')
     


def rotate(image_path, degrees_to_rotate, saved_location):
    """
    Rotate the given photo the amount of given degreesk, show it and save it
 
    @param image_path: The path to the image to edit
    @param degrees_to_rotate: The number of degrees to rotate the image
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    rotated_image = image_obj.rotate(degrees_to_rotate)
    rotated_image.save(saved_location)
    rotated_image.show()

def rotate_2():
     img = cv2.imread('test.png',2)
     cv2.imshow('',img)
     cv2.waitKey(0)

     img90 = np.rot90(img)
     cv2.imshow('',img90)
     cv2.waitKey(0)
     
     
def flip_image(image_path, saved_location):
    """
    Flip or mirror the image
 
    @param image_path: The path to the image to edit
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    rotated_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
    rotated_image.save(saved_location)
    rotated_image.show()
    
def noisecancel():
     #noise cancellation
     
     # load color image
     im = cv2.imread('test.png')

     # smooth the image with alternative closing and opening
     # with an enlarging kernel
     morph = im.copy()

     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
     morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
     morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

     # take morphological gradient
     gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)

     # split the gradient image into channels
     image_channels = np.split(np.asarray(gradient_image), 3, axis=2)

     channel_height, channel_width, _ = image_channels[0].shape

     # apply Otsu threshold to each channel
     for i in range(0, 3):
         _, image_channels[i] = cv2.threshold(~image_channels[i], 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
         image_channels[i] = np.reshape(image_channels[i], newshape=(channel_height, channel_width, 1))

     # merge the channels
     image_channels = np.concatenate((image_channels[0], image_channels[1], image_channels[2]), axis=2)

     # save the denoised image
     noise = cv2.imwrite('Noise.png', image_channels)
     cv2.imshow('Noise.png',image_channels)
     
def boxeveryword():
     #boxing every character
     filename = 'test.png'

     # read the image and get the dimensions
     img = cv2.imread(filename)
     h, w, _ = img.shape # assumes color image

     # run tesseract, returning the bounding boxes
     boxes = pytesseract.image_to_boxes(img) # also include any config options you use

     # draw the bounding boxes on the image
     for b in boxes.splitlines():
         b = b.split(' ')
         img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

     # show annotated image and wait for keypress
     cv2.imshow(filename, img)
     cv2.waitKey(0)
     
def boundingbox():
     #Bounding Box every word or bunch of words
     
     #import image
     image = cv2.imread('test.png')
     #cv2.imshow('orig',image)
     #cv2.waitKey(0)

     #grayscale
     gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
     cv2.imshow('gray',gray)
     cv2.waitKey(0)

     #binary
     ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
     cv2.imshow('second',thresh)
     cv2.waitKey(0)

     #dilation
     kernel = np.ones((5,100), np.uint8)
     img_dilation = cv2.dilate(thresh, kernel, iterations=1)
     cv2.imshow('dilated',img_dilation)
     cv2.waitKey(0)

     #find contours
     im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

     #sort contours
     sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

     for i, ctr in enumerate(sorted_ctrs):
         # Get bounding box
         x, y, w, h = cv2.boundingRect(ctr)

         # Getting ROI
         roi = image[y:y+h, x:x+w]

         # show ROI
         cv2.imshow('segment no:'+str(i),roi)
         cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
         cv2.waitKey(0)

     cv2.imshow('marked areas',image)
     cv2.waitKey(0)
def sharp():
     #Linux window/threading setup code.
     cv2.startWindowThread()
     cv2.namedWindow("Original")
     cv2.namedWindow("Sharpen")

     #Load source / input image as grayscale, also works on color images...
     imgIn = cv2.imread("test2.png", cv2.IMREAD_GRAYSCALE)
     cv2.imshow("Original", imgIn)


     #Create the identity filter, but with the 1 shifted to the right!
     kernel = np.zeros( (9,9), np.float32)
     kernel[4,4] = 2.0   #Identity, times two! 

     #Create a box filter:
     boxFilter = np.ones( (9,9), np.float32) / 81.0

     #Subtract the two:
     kernel = kernel - boxFilter


     #Note that we are subject to overflow and underflow here...but I believe that
     # filter2D clips top and bottom ranges on the output, plus you'd need a
     # very bright or very dark pixel surrounded by the opposite type.

     custom = cv2.filter2D(imgIn, -1, kernel)
     cv2.imshow("Sharpen", custom)
     cv2.waitKey(0)
     
def binary():
     image_file = Image.open("test2.png") # open colour image
     image_file = image_file.convert('1') # convert image to black and white
     image_file.save('result.png')

def imagetotext():
     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'
     text = pytesseract.image_to_string(Image.open('test.png'))
     #text=re.sub(r'\s+', '', text)
     print(text)

def main():
    path = r"C:\Users\Gaurang\Desktop\Python\Image Analysis\All Images"
    dirs = os.listdir( path )
    # This would print all the files and directories
    for file in dirs:
        print(file)
        #boxeveryword()
        #boundingbox()
        #opencv_gray()
        #noisecancel()
        #resize()
        #rotate('test.png', 90, 'rotated_mantis.png')
        #flip_image('test.png', 'flip.png')
        #rotate_2()
        #sharp()
        #binary()
        imagetotext()
  
if __name__== "__main__":
  main()
