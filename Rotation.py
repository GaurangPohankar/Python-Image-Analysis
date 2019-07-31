import numpy as np
import cv2

img = cv2.imread('test.png',1)
cv2.imshow('',img)
cv2.waitKey(0)

img90 = np.rot90(img)
cv2.imshow('',img90)
cv2.waitKey(0)
