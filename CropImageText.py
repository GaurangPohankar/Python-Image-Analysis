import re
import os
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'



#text=re.sub(r'\s+', '', text)

img = Image.open('test3.png')
area = (400, 400, 800, 800)
cropped_img = img.crop(area)
cropped_img.show()
text = pytesseract.image_to_string(cropped_img)

print(text)
