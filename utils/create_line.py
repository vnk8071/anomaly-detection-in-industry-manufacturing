import cv2
import numpy as np

path = "data/custom/train/good/2.png"
  
image = cv2.imread(path)
w, h, c = image.shape
start_point = (int(3*h/4), int(w/3))
end_point = (int(h/4), int(w/3))

# start_point_1 = (int(0), int(3*w/4))
# end_point_1 = (int(h), int(3*w/4))

color = (255, 255, 255)
thickness = 5

mask = np.zeros((image.shape), np.uint8)
mask = cv2.line(mask, start_point, end_point, (255, 255, 255), thickness)
# mask = cv2.line(mask, start_point_1, end_point_1, (255, 255, 255), thickness)
cv2.imwrite("imgs_augment/mask/2_4_mask.png", mask)

image = cv2.line(image, start_point, end_point, color, thickness)
# image = cv2.line(image, start_point_1, end_point_1, color, thickness)
cv2.imwrite("imgs_augment/mask/2_4.png", image)