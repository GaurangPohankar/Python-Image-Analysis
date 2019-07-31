from PIL import Image
from pytesseract import image_to_string

img = Image.open('7.jpg')
text = image_to_string(img)
print (text)
