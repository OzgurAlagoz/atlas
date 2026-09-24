import argparse
import cv2
import datetime
import os
from atlas.video.source import VideoSource
from configs.config import VIDEO_RES, VIDEO_SOURCE, OUTPUT
from atlas.visualization.renderer import Renderer
from atlas.video.recorder import Recorder

def main():
    parser = argparse.ArgumentParser(prog='atlas')
    parser.add_argument('--source', default=VIDEO_SOURCE)
    parser.add_argument('--res', nargs=2, type=int, default=VIDEO_RES)
    parser.add_argument('--save', action='store_true', default=False)
    args = parser.parse_args()

    video_source = VideoSource(args.source, args.res)
    fourcc = "mp4v"

    if video_source.open():
        renderer = Renderer('atlas video footage', video_source.prop_fps, video_source.frame_count)
        recorder = None
        if args.save:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(OUTPUT, timestamp + '.mp4')
            recorder = Recorder(video_source.prop_fps, video_source.resolution, fourcc)
            if recorder.open(filename):
                print('Kayit basladi: ' + filename)
            else:
                print('Kayit baslatilamadi: ' + filename)
                recorder = None
        while True:
            frame = video_source.read()
            if frame is None:
                break
            render_check = renderer.render(frame)
            if recorder is not None:
                recorder.write_frame(frame)
            if render_check is False:
                break
        video_source.close()
        if recorder is not None:
            recorder.close()
        cv2.destroyAllWindows()
    else:
        print('Dosya yolu yanlis.')

if __name__ == '__main__':
    main()
