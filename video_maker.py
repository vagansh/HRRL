import os
import moviepy.video.io.ImageSequenceClip
from PIL import Image, ImageFile
import errno

#brew install ffmpeg when the prog does not find ffmpeg

ImageFile.LOAD_TRUNCATED_IMAGES = True

image_files = []

try:
    for img_number in range(0, 1329):
        image_files.append('images/frame_' + str(img_number) + '.png')
except IOError as e:
    if e.errno == errno.EPIPE:
        pass
        # Handling of the error

fps = 30
#print(image_files)
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile('images/a_neurips/1.mp4')
