import os 
import pathlib

####################################################################################################
# @get_files_in_directory
####################################################################################################
def get_files_in_directory(directory,
                           file_extension=None) -> list:
                           
    # A list of all the files that exist in a directory
    files = list()

    # If the extension is not specified
    if file_extension is None:
        for i_file in os.listdir(directory):
            files.append(i_file)

    # Otherwise, return files that have specific extensions
    else:
        for i_file in os.listdir(directory):
            if i_file.endswith(file_extension):
                files.append(int(i_file.strip('.%s' % file_extension)))

    # Sort to ensure that you get the consequentive frames 
    files.sort()

    # Return the list
    return files


####################################################################################################
# @create_directory
####################################################################################################
def create_directory(path):

    # if the path exists, remove it and create another one.
    if os.path.exists(path):
        return
    try:
        os.mkdir(path)
    except ValueError:
        print('ERROR: cannot create directory %s' % path)


####################################################################################################
# @get_videos_list
####################################################################################################
def get_videos_list(path):

    # A lis of all the video files 
    videos = list()

    # List fll the files 
    files = os.listdir(path) 

    # Check the validity of each file 
    for f in files:

        # Get the extension 
        filename, file_extension = os.path.splitext(f)
        file_extension.lower()

        # If a valid extension, append it to the list
        if 'avi' in file_extension.lower() or \
           'tif' in file_extension.lower() or \
           'mp4' in file_extension.lower():
           videos.append(f)

    # Return the list 
    return videos 


####################################################################################################
# @veryify_input_options
####################################################################################################
def veryify_input_options(video_sequence, 
                          output_directory, 
                          pixel_threshold):

    # Verification if the video file exists 
    if not os.path.exists(video_sequence):
        print('ERROR: The video file [%s] does NOT exist. CANNOT PROCEED SUCCESSFULLY!' % video_sequence)

    try: 
        os.mkdir(output_directory)
    except: 
        print('NOTE: The output path [%s] exists'% output_directory)

    if not os.path.exists(output_directory):
        print('ERROR: The output directory [%s] does NOT exist. CANNOT PROCEED SUCCESSFULLY!' % video_sequence)
        return 
    
    # Create an output-directory that is specific to the input sequence 
    specific_output_directory = "%s/%s" % (output_directory, pathlib.Path(video_sequence).stem) 

    try: 
        os.mkdir(specific_output_directory)
    except: 
        pass 
    
    # Return the output directory 
    return specific_output_directory