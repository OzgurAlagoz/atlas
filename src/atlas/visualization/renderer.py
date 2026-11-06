import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS = 1
LINE_HEIGHT = 24
MARGIN = 10
TEXT_COLOR = (255, 255, 255)
BOX_COLOR = (0, 0, 0)


class Renderer():
    def __init__(self, window_name, prop_fps, frame_count):
        self.window_name = window_name
        self.prop_fps = prop_fps
        self.frame_count = int(frame_count)

    def build_lines(self, frame, frame_number, stats=None):
        height, width = frame.shape[:2]

        if self.frame_count > 0:
            frame_text = f"Frame: {frame_number}/{self.frame_count}"
        else:
            frame_text = f"Frame: {frame_number}"

        if stats is None or stats.ema is None:
            fps_text = "--"
        else:
            fps_text = f"{stats.ema:.0f}"

        if stats is None or stats.latency_ms is None:
            latency_text = "--"
        else:
            latency_text = f"{stats.latency_ms:.1f} ms"

        return [
            frame_text,
            f"Input FPS: {self.prop_fps:.0f}",
            f"Processing FPS: {fps_text}",
            f"Latency: {latency_text}",
            f"Resolution: {width}x{height}",
        ]

    def render(self, frame, frame_number, stats=None):
        lines = self.build_lines(frame, frame_number, stats)

        text_width = max(cv2.getTextSize(line, FONT, FONT_SCALE, THICKNESS)[0][0]
                         for line in lines)
        box_width = text_width + 2 * MARGIN
        box_height = len(lines) * LINE_HEIGHT + MARGIN
        cv2.rectangle(frame, (0, 0), (box_width, box_height), BOX_COLOR, -1)

        for index, line in enumerate(lines):
            y = MARGIN + (index + 1) * LINE_HEIGHT - 6
            cv2.putText(frame, line, (MARGIN, y), FONT, FONT_SCALE,
                        TEXT_COLOR, THICKNESS, cv2.LINE_AA)

        return frame

    def show(self, frame):
        cv2.imshow(self.window_name, frame)
        return cv2.waitKey(1) & 0xFF

    def close(self):
        cv2.destroyAllWindows()
