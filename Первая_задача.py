import cv2
import numpy as np

# We Load an image from file
img = cv2.imread('images/variant-4.jpeg')

# Now we should split the image into its color channels
b, g, r = cv2.split(img)

# Here we create a new image by merging the blue channel
bg_img = cv2.merge([b, np.zeros_like(b), np.zeros_like(b)])

# Now this is the display in a window
cv2.imshow("Blue Image", bg_img)
cv2.waitKey(0)
cv2.destroyAllWindows()