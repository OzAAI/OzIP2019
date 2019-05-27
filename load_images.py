import numpy as np
import cv2

def resize_to_image(pixels,height,width):
    '''
    load_data_as_image receives a column of strings, each string containing pixel
        values for an image, and transforms it into a matrices of correct shapes
    It also requires the width and height of the expected output shape.    
    '''
    pixels = pixels.tolist()
    faces = []
    # defining img dimensions
    image_size=(height, width)
    # resizing the string sequence into 48 x 48 integer arrays
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(height, width)
        face = cv2.resize(face.astype('uint8'),image_size)
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    # we don't have 3 channels (RGB) since we work with grayscale images
    # but we do need to add the 3rd depth channel
    faces = np.expand_dims(faces, -1)
    return faces