'''Source: 
https://github.com/whynotw/rotational-data-augmentation-yolo
https://learnopencv.com/image-rotation-and-translation-using-opencv/
'''

import numpy as np
import cv2
from glob import glob
import os
from tqdm import tqdm
import albumentations as A
import cv2

# Params
num_images = 25
input_path = './data/aqa/train/good/8.png'
output_path = './imgs_augment/'

def main():
    # Declare an augmentation pipeline
    transform = A.Compose([
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.RandomGamma(p=0.5),
        A.Rotate(limit=3, p=0.5),
        A.ChannelShuffle(p=0.5),
        A.RGBShift(p=0.5),
    ])

    # Read an image with OpenCV and convert it to the RGB colorspace
    image = cv2.imread(input_path)
    image_name = input_path.rsplit('/')[-1].split('.')[0]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for i in tqdm(range(num_images)):
        # Augment an image
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        cv2.imwrite(f"{output_path}{image_name}_{i}.png", transformed_image)


if __name__ == '__main__':
    main()
    print('DONE')