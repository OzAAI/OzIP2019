import numpy as np
from PIL import Image as im

def string_to_face(pixels,width,height):
    '''
    string_to_face receives a column of strings, each string containing pixel
        values for an image, and transforms it back into an image.
    It also requires the width and height of the expected output image.    
    '''
    faces=[]
#    we iterate for each row
    for i in range(len(pixels)):
#        splitting the long string into individual pixel values
        img = pixels[i].split()
#        creating a numpy array and changint its data type to integers
        img = np.asarray(img,dtype=np.int)
#        reshaping the array into a table with a value for each pixel
        img = img.reshape(width,height)
#        converting value array to image
        img = im.fromarray(img.astype('uint8'))
#        adding the image to the final array of images to output
        faces.append(img)
    return faces



#fer_dir = 'C:/Users/franco.ferrero/Documents/Datasets/fer2013.csv'
#
#import pandas as pd
#fer_dataset = pd.read_csv(fer_dir)
##print(fer_dataset.head())
##print(fer_dataset[0:1]['pixels'].str.split())
#
#faces = string_to_face(fer_dataset['pixels'],48,48)
#faces[26].show()