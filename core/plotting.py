import numpy 
import random 
import cv2
import os
from PIL import Image
from matplotlib import pyplot 
from matplotlib import pyplot
import matplotlib.pyplot as pyplot
import matplotlib.font_manager as font_manager
from matplotlib.ticker import FuncFormatter
import seaborn

####################################################################################################
# @verify_plotting_packages
####################################################################################################
def verify_plotting_packages():

    # Import the fonts
    font_dirs = list()
    font_dirs.extend([os.path.dirname(os.path.realpath(__file__)) + '/../fonts/'])
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)


####################################################################################################
# @plot_trajectories_on_frame
####################################################################################################
def plot_trajectories_on_frame(frame, trajectories, output_path):

    # Create an RGB image from the input frame 
    rgb_image = Image.fromarray(frame).convert("RGB")
    
    # Create a numpy array from the image 
    np_image = numpy.array(rgb_image)

    # Draw each trajectory 
    for i, trajectory in enumerate(trajectories):
        
        # Create random colors 
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)

        # Starting pixel 
        cv2.circle(np_image, (int(trajectory[0][1]), int(trajectory[0][0])), 1, (r,g,b), 1)

        # The rest of the trajectory 
        for kk in range(len(trajectory) - 1):
            
            # First point 
            y0 = int(trajectory[kk][0])
            x0 = int(trajectory[kk][1])

            # Last point 
            y1 = int(trajectory[kk + 1][0])
            x1 = int(trajectory[kk + 1][1])

            # Create the line 
            cv2.line(np_image, (x0,y0), (x1,y1), (r,g,b), 1)
    
    # Save the trajectory image 
    cv2.imwrite('%s.png' % output_path, np_image)


####################################################################################################
# @plot_trajectories
####################################################################################################
def plot_trajectories(size, trajectories, output_path):

    # Create an RGB image from the input frame 
    rgb_image = Image.new(mode="RGB", size=size)
    
    # Create a numpy array from the image 
    np_image = numpy.array(rgb_image)

    # Draw each trajectory 

    for i, trajectory in enumerate(trajectories):
        
        # Create random colors 
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)

        # Starting pixel 
        cv2.circle(np_image, (int(trajectory[0][1]), int(trajectory[0][0])), 1, (r,g,b), 1)

        # The rest of the trajectory 
        for kk in range(len(trajectory) - 1):
            
            # First point 
            x0 = int(trajectory[kk][0])
            y0 = int(trajectory[kk][1])

            # Last point 
            x1 = int(trajectory[kk + 1][0])
            y1 = int(trajectory[kk + 1][1])

            # Create the line 
            cv2.line(np_image, (x0,y0), (x1,y1), (r,g,b), 1)
    
    # Save the trajectory image 
    cv2.imwrite('%s.png' % output_path, np_image)


def plot_frame(frame, output_directory, frame_prefix):

    verify_plotting_packages()
    


    fig, ax = pyplot.subplots()
    ax.imshow(frame)
    pyplot.savefig('/projects/hidpy/output2/xx.png', dpi=300, bbox_inches='tight')

    pass 

