import argparse
import cv2
from atlas.video.source import VideoSource
from configs.config import VIDEO_RES, VIDEO_SOURCE
from atlas.visualization.renderer import Renderer

def main():
    parser = argparse.ArgumentParser(prog='atlas')
    parser.add_argument('--source', default=VIDEO_SOURCE)
    parser.add_argument('--res', nargs=2, type=int, default=VIDEO_RES)
    args = parser.parse_args()

    video_source = VideoSource(args.source, args.res)

    if video_source.open():
        renderer = Renderer('atlas video footage', video_source.prop_fps, video_source.frame_count)
        while True:
            frame = video_source.read()
            if frame is None:
                break
            render_check = renderer.render(frame)
            if render_check is False:
                break
        video_source.close()
        cv2.destroyAllWindows()
    else:
        print('Dosya yolu yanlis.')

if __name__ == '__main__':
    main()
