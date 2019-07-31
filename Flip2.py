from PIL import Image
 
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
 
if __name__ == '__main__':
    image = 'test.png'
    flip_image(image, 'flip.png')
