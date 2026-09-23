import cv2
import time

class Renderer():
    def __init__(self, window_name, prop_fps, frame_count):
        self.window_name = window_name
        self.prop_fps = prop_fps
        self.frame_count = frame_count
        self.counter = 0
        self.time = time.perf_counter()

    def render(self, frame):
           
        self.counter += 1
        self.current_frame_time = time.perf_counter()

        cv2.putText(frame, 'PROP FPS: ' + str(self.prop_fps), (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0 ,0), 1)
        cv2.putText(frame, 'TOTAL FRAME: ' + str(self.frame_count), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0 ,0), 1)
        cv2.putText(frame, 'AVG FPS: ' + str(self.counter / (self.current_frame_time-self.time)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0 ,0), 1)
        cv2.imshow(self.window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        else:
            return True

