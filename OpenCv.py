import cv2
 
image = cv2.imread("test2.png")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)
cv2.imshow("Original- gray", gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
