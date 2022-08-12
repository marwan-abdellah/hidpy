import numpy 
import time 
from PIL import Image

# CHANGE THIS WITH YOUR DIRECTORY
directory = '/ssd3/tiff'

dapi_image_path = 'dapi.tif'
dapi_image = Image.open('%s/%s' % (directory, dapi_image_path))
dapi_width, dapi_height = dapi_image.size

opal_image_path = 'opal.tif'
opal_image = Image.open('%s/%s' % (directory, opal_image_path))
opal_width, opal_height = opal_image.size

for threshold in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]:
    t0 = time.time()
    result = numpy.transpose(numpy.zeros_like(dapi_image))
    for ii in range(dapi_width):
        for jj in range(dapi_height):
            if opal_image.getpixel((ii, jj)) > threshold:
                if dapi_image.getpixel((ii, jj)) > 0.0:
                    result[ii, jj] = dapi_image.getpixel((ii, jj)) 

    result_image = Image.fromarray(result.transpose())
    result_image.save('%s/result-%s.tif' % (directory, str(threshold)))
    result_image.close()
    t1 = time.time()

dapi_image.close()
opal_image.close()