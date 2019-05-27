import numpy as np
import cv2

def load_data_as_image(pixels,height,width):
    '''
    load_data_as_image receives a column of strings, each string containing pixel
        values for an image, and transforms it into a matrices of correct shapes
    It also requires the width and height of the expected output shape.    
    '''
#    faces=[]
#    for i in range(len(pixels)):
##        splitting the long string into individual pixel values
#        img = pixels[i].split()
##        creating a numpy array and changint its data type to integers
#        img = np.asarray(img,dtype=np.int)
##        reshaping the array into a table with a value for each pixel
#        img = img.reshape(height,width,1)
##        adding the image to the final array of images to output
#        faces.append(img)
#    return faces
    pixels = pixels[:][0].tolist()
    faces = []
    image_size=(height, width)
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(height, width)
        face = cv2.resize(face.astype('uint8'),image_size)
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    return faces