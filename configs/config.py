import os

path = __file__
parent_path = os.path.dirname(os.path.dirname(path))
video = os.path.join(parent_path, 'datasets', 'videos', 'video_simple_room.mp4')

VIDEO_SOURCE = video
VIDEO_RES = 780, 540