import cv2
 
# Read the image.
img = cv2.imread('taj.jpeg')
 
# Apply bilateral filter with d = 15,
# sigmaColor = sigmaSpace = 75.
bilateral = cv2.bilateralFilter(img, 15, 75, 75)
 
# Save the output.
cv2.imwrite('taj_bilateral.jpg', bilateral)

# Read the image.
img = cv2.imread('sequence1.jpeg')
 
# Apply bilateral filter with d = 15,
# sigmaColor = sigmaSpace = 75.
bilateral = cv2.bilateralFilter(img, 15, 75, 75)
 
# Save the output.
cv2.imwrite('sequence1_bilateral.jpg', bilateral)
