import cv2
import numpy as np

img = np.zeros((400, 600, 3), dtype=np.uint8)

# Red rectangle
cv2.rectangle(img, (50, 50), (200, 150), (0, 0, 255), -1)

# Green circle
cv2.circle(img, (400, 100), 70, (0, 255, 0), -1)

# Blue triangle
pts = np.array([[300, 350], [200, 250], [400, 250]], np.int32)
cv2.fillPoly(img, [pts], (255, 0, 0))

# White text
cv2.putText(img, "Test Image", (180, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imwrite("output/test_image.jpg", img)
print("Created output/test_image.jpg")
