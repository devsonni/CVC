import cv2 as cv
import numpy as np

# Loading exposure images into a list
img_fn = ["IMAGE_1.JPG", "IMAGE_2.JPG", "IMAGE_3.JPG"]
img_list = [cv.imread(fn) for fn in img_fn]
exposure_times = np.array([15.0, 2.5, 0.25, 0.0333], dtype=np.float32)

# Exposure fusion using Mertens
merge_mertens = cv.createMergeMertens()
res_mertens = merge_mertens.process(img_list)

# converting back to the 8-bit image
res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')

# saving the file
cv.imwrite("fusion_mertens.jpg", res_mertens_8bit)