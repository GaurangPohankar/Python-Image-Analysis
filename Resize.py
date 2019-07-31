from PIL import Image

img = Image.open('test.png') # image extension *.png,*.jpg
new_width  = 200
new_height = 300
img = img.resize((new_width, new_height), Image.ANTIALIAS)
img.save('resized.png')
