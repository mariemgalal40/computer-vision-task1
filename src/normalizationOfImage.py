from PIL import Image, ImageOps
from numpy import asarray

def normalize_color(image):
        new_min = 0
        new_max = 255

        # get pixels values and knowing max and min value
        pixels = asarray(image)
        pixels = pixels.astype('float32')
        input_min = pixels.min()
        input_max = pixels.max()

        # specify scale to be from 0 to 255
        scaling = ((new_max - new_min)/(input_max - input_min))

        # pixel mapping
        
        height, width = image.size
        new_image = Image.new(mode="RGBA", size=(height, width))
        pixel_map = new_image.load()

        for x in range(height):
            for y in range(width):
                r, g, b = image.getpixel((x, y))
                pixel_map[x, y] = (int(r * scaling), int(g * scaling), int(b * scaling))

        return new_image

