import numpy 
import random 
import cv2
import os
from PIL import Image
import matplotlib
from matplotlib import pyplot 
from matplotlib import colors
import matplotlib.pyplot as pyplot
import matplotlib.font_manager as font_manager
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
import seaborn
import time


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
# @sample_range
####################################################################################################
def sample_range(start,
                 end,
                 steps):

    # Delta
    delta = 1. * (end - start) / (steps - 1)

    # Data
    data = list()
    for i in range(steps):
        value = start + i * delta
        data.append(value)

    return data


####################################################################################################
# @plot_trajectories_on_frame
####################################################################################################
def plot_trajectories_on_frame(frame, trajectories, output_path):

    # Compute the time 
    start = time.time()
    
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


####################################################################################################
# @plot_frame
####################################################################################################
def plot_frame(frame, output_directory, frame_prefix, font_size=10, tick_count=5):

    verify_plotting_packages()
    
    seaborn.set_style("whitegrid")
    pyplot.rcParams['axes.grid'] = 'True'
    pyplot.rcParams['grid.linewidth'] = 0.5
    pyplot.rcParams['grid.color'] = 'black'
    pyplot.rcParams['grid.alpha'] = 0.25
    pyplot.rcParams['font.family'] = 'NimbusSanL'
    pyplot.rcParams['font.monospace'] = 'Regular'
    pyplot.rcParams['font.style'] = 'normal'
    pyplot.rcParams['axes.labelweight'] = 'light'
    pyplot.rcParams['axes.linewidth'] = 0.5
    pyplot.rcParams['axes.labelsize'] = font_size
    pyplot.rcParams['xtick.labelsize'] = font_size
    pyplot.rcParams['ytick.labelsize'] = font_size
    pyplot.rcParams['legend.fontsize'] = font_size
    pyplot.rcParams['figure.titlesize'] = font_size
    pyplot.rcParams['axes.titlesize'] = font_size
    pyplot.rcParams['xtick.major.pad'] = '1'
    pyplot.rcParams['ytick.major.pad'] = '1'
    pyplot.rcParams['axes.edgecolor'] = '0'
    pyplot.rcParams['axes.autolimit_mode'] = 'round_numbers'


    # Plot 
    fig, ax = pyplot.subplots()
    
    # Create the ticks of the images 
    xticks = sample_range(0, frame.shape[0], tick_count)
    yticks = sample_range(0, frame.shape[1], tick_count)

    # Show the image 
    im = pyplot.imshow(frame)
    ax.set_xlim(xticks[0], xticks[-1])
    ax.set_ylim(yticks[0], yticks[-1])
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    
    
    # Color-basr axis 
    cax = ax.inset_axes([0.00, -0.15, 1.0, 0.05])

    # Create the ticks based on the range 
    cbticks = sample_range((frame.min()), (frame.max()), 4)
    cbticks = list(map(int, cbticks))

    # Convert the ticks to a numpy array 
    cbticks = numpy.array(cbticks)
    
    # Color-bar 
    cb = pyplot.colorbar(im, ax=ax, cax=cax, orientation="horizontal", ticks=cbticks)
    cb.ax.tick_params(labelsize=font_size, width=0.5) 
    cb.ax.set_xlim((cbticks[0], cbticks[-1]))
    cb.update_ticks()

    # Save the figure 
    pyplot.savefig('%s/%s.png' % (output_directory, frame_prefix), dpi=300, bbox_inches='tight', pad_inches=0)


####################################################################################################
# @plot_frame
####################################################################################################
def plot_labels_map(labels_map, output_directory, frame_prefix, font_size=10, npop=1):


    from matplotlib import colors, pyplot
    import seaborn

    import numpy
    seaborn.set_style("whitegrid")
    pyplot.rcParams['axes.grid'] = 'True'
    pyplot.rcParams['grid.linewidth'] = 0.5
    pyplot.rcParams['grid.color'] = 'black'
    pyplot.rcParams['grid.alpha'] = 0.25
    pyplot.rcParams['font.family'] = 'NimbusSanL'
    pyplot.rcParams['font.monospace'] = 'Regular'
    pyplot.rcParams['font.style'] = 'normal'
    pyplot.rcParams['axes.labelweight'] = 'light'
    pyplot.rcParams['axes.linewidth'] = 0.5
    pyplot.rcParams['axes.labelsize'] = font_size
    pyplot.rcParams['xtick.labelsize'] = font_size
    pyplot.rcParams['ytick.labelsize'] = font_size
    pyplot.rcParams['legend.fontsize'] = font_size
    pyplot.rcParams['figure.titlesize'] = font_size
    pyplot.rcParams['axes.titlesize'] = font_size
    pyplot.rcParams['xtick.major.pad'] = '1'
    pyplot.rcParams['ytick.major.pad'] = '1'
    pyplot.rcParams['axes.edgecolor'] = '0'
    pyplot.rcParams['axes.autolimit_mode'] = 'round_numbers'

    bounds = [0.5, 1.5, 2.5, 3.5, 4.5]

    pyplot.clf

    # Plot 
    fig, ax = pyplot.subplots()
    
    # Create the ticks of the images 
    xticks = sample_range(0, labels_map.shape[0], 5)
    yticks = sample_range(0, labels_map.shape[1], 5)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    

    listcolors=['w','g','b','purple','r','greenyellow']
    cmap = colors.ListedColormap(listcolors[0:npop+1])

    img1=ax.imshow(labels_map, interpolation='nearest',cmap=cmap,origin='lower')
    # Show the image 
    
     # Color-basr axis 
    cax = ax.inset_axes([0.00, -0.15, 1.0, 0.05])

    cbar=fig.colorbar(img1, ax=ax,spacing='proportional',orientation='horizontal',boundaries=[-0.5] + bounds[0:npop+1] + [npop+0.5], cax=cax)
    cbar.outline.set_color('black')
    cbar.outline.set_linewidth(1)
    labels_cbar = numpy.arange(0, npop+1, 1)
    loc = labels_cbar
    cbar.set_ticks(loc)


    # Save the figure 
    pyplot.savefig('%s/%s.png' % (output_directory, frame_prefix), dpi=300, bbox_inches='tight', pad_inches=0)


####################################################################################################
# @plot_model_selection_image
####################################################################################################
def plot_model_selection_image(model_selection_matrix, 
                               mask_matrix, 
                               output_directory, 
                               frame_prefix, 
                               font_size=14, 
                               title='Model Selection', 
                               tick_count=5):

    verify_plotting_packages()

    # Styles 
    seaborn.set_style("whitegrid")
    pyplot.rcParams['axes.grid'] = 'False'
    pyplot.rcParams['grid.linewidth'] = 0.5
    pyplot.rcParams['grid.color'] = 'black'
    pyplot.rcParams['grid.alpha'] = 0.25
    pyplot.rcParams['font.family'] = 'NimbusSanL'
    pyplot.rcParams['font.monospace'] = 'Regular'
    pyplot.rcParams['font.style'] = 'normal'
    pyplot.rcParams['axes.labelweight'] = 'light'
    pyplot.rcParams['axes.linewidth'] = 0.5
    pyplot.rcParams['axes.labelsize'] = font_size
    pyplot.rcParams['xtick.labelsize'] = font_size
    pyplot.rcParams['ytick.labelsize'] = font_size
    pyplot.rcParams['legend.fontsize'] = font_size
    pyplot.rcParams['figure.titlesize'] = font_size
    pyplot.rcParams['axes.titlesize'] = font_size
    pyplot.rcParams['xtick.major.pad'] = '1'
    pyplot.rcParams['ytick.major.pad'] = '1'
    pyplot.rcParams['axes.edgecolor'] = '0'

    # A new figure 
    pyplot.clf
    fig, ax = pyplot.subplots()

    # Create the color-map 
    palette = seaborn.color_palette("hls", 5)
    palette.insert(0, 'w')
    cmap = colors.ListedColormap(palette)
    
    # Render the image 
    image = ax.imshow(model_selection_matrix, interpolation='nearest', cmap=cmap, origin='lower')
    ax.contour(mask_matrix, colors='k', origin='lower')

    # Create the ticks of the images 
    xticks = sample_range(0, model_selection_matrix.shape[0], tick_count)
    yticks = sample_range(0, model_selection_matrix.shape[1], tick_count)

    # Update the axex 
    ax.set_xlim(xticks[0], xticks[-1])
    ax.set_ylim(yticks[0], yticks[-1])
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # Update the title 
    ax.set_title(title)

    # Color-bar bounds  
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5]

    cbar = fig.colorbar(image,ax=ax,spacing='proportional',orientation='vertical',boundaries=[-0.5] + bounds + [5.5])
    cbar.set_ticks(numpy.arange(0, 6, 1))
    cbar.set_ticklabels([' ','D','DA','V','DV','DAV'])

    # Save the figure 
    pyplot.savefig('%s/%s.png' % (output_directory, frame_prefix), dpi=300, bbox_inches='tight', pad_inches=0)


####################################################################################################
# @plot_matrix_map
####################################################################################################
def plot_matrix_map(matrix, mask_matrix, output_directory, frame_prefix, font_size=14, title='Matrix', tick_count=5):

    verify_plotting_packages()
    
    seaborn.set_style("whitegrid")
    pyplot.rcParams['axes.grid'] = 'False'
    pyplot.rcParams['grid.linewidth'] = 0.5
    pyplot.rcParams['grid.color'] = 'black'
    pyplot.rcParams['grid.alpha'] = 0.25
    pyplot.rcParams['font.family'] = 'NimbusSanL'
    pyplot.rcParams['font.monospace'] = 'Regular'
    pyplot.rcParams['font.style'] = 'normal'
    pyplot.rcParams['axes.labelweight'] = 'light'
    pyplot.rcParams['axes.linewidth'] = 0.5
    pyplot.rcParams['axes.labelsize'] = font_size
    pyplot.rcParams['xtick.labelsize'] = font_size
    pyplot.rcParams['ytick.labelsize'] = font_size
    pyplot.rcParams['legend.fontsize'] = font_size
    pyplot.rcParams['figure.titlesize'] = font_size
    pyplot.rcParams['axes.titlesize'] = font_size
    pyplot.rcParams['xtick.major.pad'] = '1'
    pyplot.rcParams['ytick.major.pad'] = '1'
    pyplot.rcParams['axes.edgecolor'] = '0'

    # New figure  
    pyplot.clf
    fig, ax = pyplot.subplots()
    
    # Create the ticks of the images 
    xticks = sample_range(0, matrix.shape[0], tick_count)
    yticks = sample_range(0, matrix.shape[1], tick_count)

    # Show the image 
    image = pyplot.imshow(matrix, interpolation='nearest',cmap='viridis',origin='lower')
    ax.contour(mask_matrix, colors='k', origin='lower')

    # Axes 
    xticks = list(map(int, xticks))
    yticks = list(map(int, yticks))
    ax.set_xlim(xticks[0], xticks[-1])
    ax.set_ylim(yticks[0], yticks[-1])
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # Title 
    ax.set_title(title)

    # Color-bar 
    fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cbar = fig.colorbar(image, ax=ax,  spacing='proportional',orientation='vertical', format=fmt)
    cbar.formatter.set_powerlimits((0, 0)) 
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Re-adjust the color-bar 
    cb_range = cbar.ax.get_ylim()
    cbticks = sample_range(cb_range[0], cb_range[-1], 4)
    cbticks = numpy.array(cbticks)
    cbar.update_ticks()

    # Save the figure 
    pyplot.savefig('%s/%s.png' % (output_directory, frame_prefix), dpi=300, bbox_inches='tight', pad_inches=0)