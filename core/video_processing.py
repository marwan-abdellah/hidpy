import numpy 
import cv2
from PIL import Image 
import imageio
from tqdm import tqdm


####################################################################################################
# @create_numpy_padded_image
####################################################################################################
def create_numpy_padded_image(image_path):
    
    # Create image object 
    loaded_image = imageio.imread(image_path, as_gray=True)
    
    # Image size 
    image_size = loaded_image.shape
    
    # Make a square image 
    square_size = image_size[0]
    if image_size[1] > image_size[0]:
        square_size = image_size[1]
    
    # Ensure that it is even 
    square_size = square_size if square_size % 2 == 0 else square_size + 1
    
    # Create a square image 
    square_image = Image.new(mode='L', size=(square_size, square_size), color='black')    
    square_image.paste(Image.fromarray(np.float32(loaded_image)))
    square_image = numpy.float32(square_image)
        
    # Return the square image 
    return square_image


####################################################################################################
# @write_frame_to_file
####################################################################################################
def write_frame_to_file(frame,
                        frame_prefix,
                        extension='bmp'):
    
    # Write the frame into an image 
    cv2.imwrite('%s.%s' % (frame_prefix, extension), frame)


####################################################################################################
# @load_frame_from_video
####################################################################################################
def load_frame_from_video(video_capture,
                          frame_number):
    
    # Set the video to the specific frame 
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Check if the video has frames or not and get a list 
    valid, frame = video_capture.read()
    
    # If the frame is valid, return it 
    if valid:

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Image size 
        image_size = frame.shape
        
        # Make a square image 
        square_size = image_size[0]
        if image_size[1] > image_size[0]:
            square_size = image_size[1]
        
        # Ensure that it is even 
        square_size = square_size if square_size % 2 == 0 else square_size + 1

        # Create a square image 
        square_image = Image.new(mode='L', size=(square_size, square_size), color='black')    
        square_image.paste(Image.fromarray(numpy.float32(frame)))
        square_image = numpy.float32(square_image)
        return square_image

    else:
        print('Invalid frame [%d]' % frame_number)
        exit(0)


####################################################################################################
# @get_frames_list_from_video
####################################################################################################
def get_frames_list_from_video(video_path,
                               verbose=False):

    # Get the video capture from the video file 
    video_capture = cv2.VideoCapture(video_path)

    # Get the frame rate 
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Get the number of frames in the video 
    number_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Print the video details 
    if verbose:
        string = "\t* Video Details: \n"
        string += "  \t* Name: %s \n" % video_path
        string += "  \t* Number Frames %d" % number_frames 
        string += "  \t* FPS: %f" % float(fps)
        print(string)

    # Save the frames to a list 
    frames = list() 
    for i in tqdm(range(number_frames), bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):
        frames.append(load_frame_from_video(video_capture=video_capture, frame_number=i))
    
    # Return the list 
    return frames
