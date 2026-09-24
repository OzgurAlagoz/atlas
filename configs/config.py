import os

path = __file__
parent_path = os.path.dirname(os.path.dirname(path))
video = os.path.join(parent_path, 'datasets', 'videos', 'video_simple_room.mp4')
outputs = os.path.join(parent_path, 'outputs')

OUTPUT = outputs
VIDEO_SOURCE = video
VIDEO_RES = 780, 540
