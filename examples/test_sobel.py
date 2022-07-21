import cv2

ddept=cv2.CV_16S

img = cv2.imread('/home/abdellah/optical-flow-trials/pyoptflow/src/pyoptflow/tests/data/sequence/sequence1-2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)






x = cv2.Sobel(gray, ddept, 1,0, ksize=3, scale=1)
y = cv2.Sobel(gray, ddept, 0,1, ksize=3, scale=1)
absx= cv2.convertScaleAbs(x)
absy = cv2.convertScaleAbs(y)
edge = cv2.addWeighted(absx, 0.5, absy, 0.5,0)
#cv2.imshow('edge', edge)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

cv2.imwrite('/home/abdellah/optical-flow-trials/pyoptflow/src/pyoptflow/tests/data/sequence/sequence-sobel-2.bmp', edge)