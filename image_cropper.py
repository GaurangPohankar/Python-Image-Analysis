import re
from PIL import Image
from pytesseract import image_to_string

img = Image.open('1539775291.5273597.png')
text = image_to_string(img)
print(text)
mob = [int(s) for s in re.findall(r'-?\d+\.?\d*', text)]
print(mob)
'''
text='3191915242448,9'
try:
    if text[0]==('4'):
        text=text[1:]
        print(text)
    elif text[0]+text[1]==('31'):
        text=text[2:]
        print(text)
except:
    text=text
    print(text)


'''
