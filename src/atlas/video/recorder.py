import cv2
import os

class Recorder:
    def __init__(self, fps, res, fourcc):
        self.fps = fps
        self.res = res
        self.fourcc = fourcc
        self.out = None

    def open(self, filename):
        self.filename = filename
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fourcc_code = cv2.VideoWriter_fourcc(*self.fourcc)
        self.out = cv2.VideoWriter(filename, fourcc_code, self.fps, self.res)
        return self.out.isOpened()

    def write_frame(self, frame):
        self.out.write(frame)

    def close(self):
        self.out.release()
