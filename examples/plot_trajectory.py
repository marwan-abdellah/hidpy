from pickletools import pylist
import cv2
import numpy as np
import argparse
import time
from sys import stdout
import matplotlib.pyplot as plt


####################################################################################################
# @parse_command_line_arguments
####################################################################################################
def parse_command_line_arguments():

    # add all the options
    description = 'This application takes an input sequence and computes the intensity profile '
    'along the entire sequence'
    parser = argparse.ArgumentParser(description=description)

    arg_help = 'Input sequence'
    parser.add_argument('--input-sequence', '-i', action='store', help=arg_help)

    arg_help = 'Output directory, where the final results/artefacts will be stored'
    parser.add_argument('--output-directory', '-o', action='store', help=arg_help)

    # Parse the arguments
    return parser.parse_args()



####################################################################################################
# @Main
####################################################################################################
if __name__ == "__main__":

    # Parse the command line arguments
    args = parse_command_line_arguments()

import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt

'''
def vector_to_rgb(angle, absolute):
    """Get the rgb value for the given `angle` and the `absolute` value

    Parameters
    ----------
    angle : float
        The angle in radians
    absolute : float
        The absolute value of the gradient
    
    Returns
    -------
    array_like
        The rgb value as a tuple with values [0..1]
    """
    global max_abs

    # normalize angle
    angle = angle % (2 * np.pi)
    if angle < 0:
        angle += 2 * np.pi

    return matplotlib.colors.hsv_to_rgb((angle / 2 / np.pi, 
                                         absolute / max_abs, 
                                         absolute / max_abs))



X = np.arange(-10, 10 + 1, 1)
Y = np.arange(-10, 10 + 1, 1)

print(X)



U, V = np.meshgrid(X, Y)


angles = np.arctan2(V, U)
lengths = np.sqrt(np.square(U) + np.square(V))

max_abs = np.max(lengths)
c = np.array(list(map(vector_to_rgb, angles.flatten(), lengths.flatten())))

fig, ax = plt.subplots()
q = ax.quiver(X, Y, U, V, color=c)

plt.savefig('sample.png')
'''



import numpy as np
import matplotlib.pyplot as plt
 
 
# Creating arrow
x_pos = 0
y_pos = 0
x_direct = 1
y_direct = 1
 
# Creating plot
x = np.arange(0,2.2,0.2)
y = np.arange(0,2.2,0.2)

X, Y = np.meshgrid(x, y)
u = np.cos(X)*Y
v = np.sin(y)*Y

fig, ax = plt.subplots(figsize=(7,7))
ax.quiver(X,Y,u,v)

# Show plot
plt.show()



