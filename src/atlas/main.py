import argparse
import datetime
import os

import cv2

from atlas.common.run_log import RunLogger, collect_environment
from atlas.common.timing import Timing
from atlas.video.recorder import Recorder
from atlas.video.source import VideoSource
from atlas.visualization.renderer import Renderer
from configs.config import OUTPUT, VIDEO_RES, VIDEO_SOURCE

PROJECT_ROOT = os.path.dirname(OUTPUT)
SNAPSHOT_DIR = os.path.join(OUTPUT, 'snapshots')


def timestamp():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def save_snapshot(frame, frame_number, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{timestamp()}_frame{frame_number:05d}.png")
    if not cv2.imwrite(path, frame):
        raise RuntimeError(f"Snapshot could not be written to: {path}")
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog='atlas')
    parser.add_argument('--source', default=VIDEO_SOURCE,
                        help='video file or camera index')
    parser.add_argument('--res', nargs=2, type=int, default=VIDEO_RES,
                        metavar=('WIDTH', 'HEIGHT'),
                        help='size the frames are resized to')
    parser.add_argument('--save', action='store_true', default=False,
                        help='record the annotated video to outputs/')
    parser.add_argument('--no-display', action='store_true', default=False,
                        help='run without opening a window')
    parser.add_argument('--run-dir', default=None,
                        help='directory for frames.csv and run.json')
    parser.add_argument('--warmup', type=int, default=5,
                        help='frames left out of the run summary')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    video_source = VideoSource(args.source, args.res)
    if not video_source.open():
        print('Dosya yolu yanlis.')
        return

    renderer = Renderer('atlas video footage', video_source.prop_fps, video_source.frame_count)

    recorder = None
    if args.save:
        filename = os.path.join(OUTPUT, timestamp() + '.mp4')
        recorder = Recorder(video_source.prop_fps, video_source.resolution, "mp4v")
        if recorder.open(filename):
            print('Kayit basladi: ' + filename)
        else:
            print('Kayit baslatilamadi: ' + filename)
            recorder = None

    run_dir = args.run_dir or os.path.join(OUTPUT, 'runs', timestamp())
    metadata = {
        'source': args.source,
        'resolution': list(args.res),
        'display': not args.no_display,
        'save': args.save,
        **collect_environment(PROJECT_ROOT),
    }
    logger = RunLogger(run_dir, metadata, warmup=args.warmup)

    timing = Timing()
    stats = None
    try:
        while True:
            timing.start_frame()
            with timing.stage('read'):
                frame = video_source.read()
            if frame is None:
                break

            frame_number = timing.counter + 1
            with timing.stage('render'):
                renderer.render(frame, frame_number, stats)

            if recorder is not None:
                with timing.stage('write'):
                    recorder.write_frame(frame)

            key = None
            if not args.no_display:
                with timing.stage('display'):
                    key = renderer.show(frame)

            stats = timing.finish_frame()
            logger.log(stats)

            if key == ord('s'):
                print('Kare kaydedildi: ' + save_snapshot(frame, frame_number, SNAPSHOT_DIR))
            if key == ord('q'):
                break
    finally:
        video_source.close()
        if recorder is not None:
            recorder.close()
        if not args.no_display:
            renderer.close()
        summary = logger.close()

    print(f"{summary['frames']} kare | "
          f"latency medyan {summary['latency_median_ms']} ms, p95 {summary['latency_p95_ms']} ms | "
          f"fps {summary['fps_mean']} | log: {run_dir}")


if __name__ == '__main__':
    main()
