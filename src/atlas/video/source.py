import cv2 

class VideoSource():
    def __init__(self, source, res):
        self.cap = None
        self.source = source
        self.resolution = res

    def open(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            return False
        else:
            self.prop_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            return True

    def read(self):
        ret, frame = self.cap.read()
        if not ret: 
            print(self.source)
            print("Video ended or reading error.")
            self.cap.release()
            return
        
        frame = cv2.resize(frame, self.resolution)
        return frame

    def close(self):
        self.cap.release()




