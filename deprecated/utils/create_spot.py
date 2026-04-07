import cv2
import numpy as np

path = "data/custom/train/good/2.png"
  
image = cv2.imread(path)
w, h, c = image.shape
point_1 = (int(h/2), int(w/2))
point_2 = (int(h/2+50), int(w/2+50))

point_3 = (int(h/2+50), int(w/2-50))
point_4 = (int(h/2-50), int(w/2+50))
point_5 = (int(h/2-50), int(w/2-50))

color = (255, 255, 255)
radius = 30
thickness = -1

mask = np.zeros((image.shape), np.uint8)
mask = cv2.circle(mask, point_1, radius, (255, 255, 255), thickness)
mask = cv2.circle(mask, point_2, radius, (255, 255, 255), thickness)
mask = cv2.circle(mask, point_3, radius, (255, 255, 255), thickness)
mask = cv2.circle(mask, point_4, radius, (255, 255, 255), thickness)
mask = cv2.circle(mask, point_5, radius, (255, 255, 255), thickness)
cv2.imwrite("imgs_augment/mask/2_2_mask.png", mask)

image = cv2.circle(image, point_1, radius, color, thickness)
image = cv2.circle(image, point_2, radius, color, thickness)
image = cv2.circle(image, point_3, radius, color, thickness)
image = cv2.circle(image, point_4, radius, color, thickness)
image = cv2.circle(image, point_5, radius, color, thickness)
cv2.imwrite("imgs_augment/mask/2_2.png", image)