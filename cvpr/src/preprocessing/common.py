"""Copyright 2022 Toyota Research Institute.  All rights reserved."""

import logging
import os

import ffmpeg
import numpy as np

logger = logging.getLogger(__name__)


predefined_eve_splits = {
    'train': ['train%02d' % i for i in range(1, 40)],
    'val': ['val%02d' % i for i in range(1, 6)],
    'test': ['test%02d' % i for i in range(1, 11)],
    'etc': ['etc%02d' % i for i in range(1, 3)],
}


def stimulus_type_from_folder_name(folder_name):
    parts = folder_name.split('_')
    if parts[1] in ('image', 'video', 'wikipedia'):
        return parts[1]
    elif parts[1] == 'eye':
        return 'points'
    raise ValueError('Given folder name unexpected: %s' % folder_name)


class VideoReader(object):
    def __init__(self, video_path, frame_indices=None, is_async=True, output_size=None, video_decoder_codec='libx264'):
        self.video_decoder_codec = video_decoder_codec
        self.is_async = is_async
        self.video_path = video_path
        self.output_size = output_size
        self.frame_indices = frame_indices
        if self.video_path.endswith('_eyes.mp4'):
            self.timestamps_path = video_path.replace('_eyes.mp4', '.timestamps.txt')
        elif self.video_path.endswith('_face.mp4'):
            self.timestamps_path = video_path.replace('_face.mp4', '.timestamps.txt')
        elif self.video_path.endswith('.128x72.mp4'):
            self.timestamps_path = video_path.replace('.128x72.mp4', '.timestamps.txt')
        else:
            self.timestamps_path = video_path.replace('.mp4', '.timestamps.txt')
        assert(os.path.isfile(self.video_path))
        assert(os.path.isfile(self.timestamps_path))

    def get_frames(self):
        assert(self.is_async is False)

        # Get frames
        self.preparations()
        input_params, output_params = self.get_params()
        buffer, _ = (
            ffmpeg.input(self.video_path, **input_params)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', loglevel="quiet",
                    **output_params)
            .run(capture_stdout=True, quiet=True)
        )
        frames = np.frombuffer(buffer, np.uint8).reshape(-1, self.height, self.width, 3)

        # Get timestamps
        timestamps = self.timestamps
        if self.frame_indices is not None:
            timestamps = self.timestamps[self.frame_indices]

        return timestamps, frames

    def preparations(self):
        # Read video file tags
        probe = ffmpeg.probe(self.video_path)
        video_stream = next((stream for stream in probe['streams']
                             if stream['codec_type'] == 'video'), None)
        self.width = video_stream['width']
        self.height = video_stream['height']
        assert self.height != 0
        assert self.width != 0
        if self.output_size is not None:
            self.width, self.height = self.output_size

        # Read timestamps file
        self.timestamps = np.loadtxt(self.timestamps_path).astype(np.int)

    def __enter__(self):
        assert(self.is_async)
        self.preparations()
        return self

    def get_params(self):
        # Input params (specifically, selection of decoder)
        input_params = {}
        if self.video_decoder_codec == 'nvdec':
            input_params = {
                'hwaccel': 'nvdec',
                'vcodec': 'h264_cuvid',
                'c:v': 'h264_cuvid',
            }
        else:
            assert(self.video_decoder_codec == 'libx264')
        input_params['vsync'] = 0

        # Set output params (resize frame here)
        output_params = {}
        if self.frame_indices is not None:
            # Index picking for range [start_index, end_index)
            # assert(len(self.frame_indices) > 1)
            cmd = 'select=\'%s\'' % '+'.join([
                ('eq(n,%d)' % index)
                for index in self.frame_indices
            ])
            output_params['vf'] = (output_params['vf'] + ',' + cmd
                                   if 'vf' in output_params else cmd)
        if self.output_size is not None:
            ow, oh = self.output_size
            cmd = 'scale=%d:%d' % (ow, oh)
            output_params['vf'] = (output_params['vf'] + ',' + cmd
                                   if 'vf' in output_params else cmd)

        return input_params, output_params

    def __iter__(self):
        assert(self.is_async)
        input_params, output_params = self.get_params()

        # Make the actual call
        self.ffmpeg_call = (
            ffmpeg
            .input(self.video_path, **input_params)
            .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel="quiet", **output_params)
            .run_async(pipe_stdout=True)
        )
        self.index = self.start_index if self.start_index is not None else 0
        return self

    def __next__(self):
        assert(self.is_async)
        in_bytes = self.ffmpeg_call.stdout.read(self.height * self.width * 3)
        if not in_bytes:
            raise StopIteration
        if self.index >= len(self.timestamps):
            raise StopIteration
        current_timestamp = self.timestamps[self.index]
        self.index += 1
        return (
            current_timestamp,
            np.frombuffer(in_bytes, dtype=np.uint8).reshape(self.height, self.width, 3)
        )

    def __exit__(self, type, value, traceback):
        if self.is_async:
            self.ffmpeg_call.stdout.close()
            self.ffmpeg_call.wait()


# Basic test to see that frames-slice grabbing works
if __name__ == '__main__':
    import argparse  # noqa
    parser = argparse.ArgumentParser(description='Merge individual videos into one.')
    parser.add_argument('video_file', type=str, help='Folder to read .mp4 files from.')
    args = parser.parse_args()

    assert(os.path.isfile(args.video_file))
    timestamps, frames = VideoReader(args.video_file, is_async=False,
                                     start_index=10, end_index=60).get_frames()
    import cv2 as cv  # noqa
    for timestamp, frame in zip(timestamps, frames):
        print(timestamp)
        cv.imshow('frame', cv.resize(frame, None, fx=0.5, fy=0.5))
        cv.waitKey(100)
