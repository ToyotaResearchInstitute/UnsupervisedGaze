"""Copyright 2022 Toyota Research Institute.  All rights reserved."""

import logging
import os
from typing import List
import random
import sys
import pathlib
from frozendict import frozendict
import bz2
import pickle
import _pickle as cPickle
from multiprocessing import Pool

import cv2 as cv
import h5py
import numpy as np

from common import stimulus_type_from_folder_name, VideoReader, predefined_eve_splits
file_dir_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(file_dir_path, ".."))
import core.training as training
from utils.data_types import MultiDict
from utils.angles import pitch_yaw_to_vector

logger = logging.getLogger(__name__)
config, device = training.script_init_common()

source_to_fps = {
    'screen': 30,
    'basler': 60,
    'webcam_l': 30,
    'webcam_c': 30,
    'webcam_r': 30,
}

source_to_interval_ms = dict([
    (source, 1e3 / fps) for source, fps in source_to_fps.items()
])

sequence_segmentations = None
cache_pkl_path = './src/segmentation_cache/10Hz_seqlen30.pkl'


class EveConfig():
    video_decoder_codec = 'libx264'  # libx264 | nvdec
    assumed_frame_rate = 10  # We will skip frames from source videos accordingly
    max_sequence_len = 30  # In frames assuming 10Hz
    face_size = [256, 256]  # width, height
    eyes_size = [128, 128]  # width, height
    screen_size = [128, 72]  # width, height
    actual_screen_size = [1920, 1080]  # DO NOT CHANGE
    camera_frame_type = 'eyes'  # full | face | eyes
    load_screen_content = False
    load_full_frame_for_visualization = False
eve_config = EveConfig()

class EveDataset():
    def __init__(self, dataset_path: str,
                 participants_to_use: List[str] = None,
                 cameras_to_use: List[str] = None,
                 types_of_stimuli: List[str] = None,
                 stimulus_name_includes: str = ''):

        if types_of_stimuli is None:
            types_of_stimuli = ['image', 'video', 'wikipedia']
        if cameras_to_use is None:
            cameras_to_use = ['basler', 'webcam_l', 'webcam_c', 'webcam_r']
        assert('points' not in types_of_stimuli)  # NOTE: deal with this in another way

        self.path = dataset_path
        self.types_of_stimuli = types_of_stimuli
        self.stimulus_name_includes = stimulus_name_includes
        self.participants_to_use = participants_to_use
        self.cameras_to_use = cameras_to_use
        
        # Some sanity checks
        assert(len(self.participants_to_use) > 0)
        assert(30 > eve_config.assumed_frame_rate)
        assert(30 % eve_config.assumed_frame_rate == 0)

        # Load or calculate sequence segmentations (start/end indices)
        global cache_pkl_path, sequence_segmentations
        cache_pkl_path = (
            './src/segmentation_cache/%dHz_seqlen%d.pkl' % (
                eve_config.assumed_frame_rate, eve_config.max_sequence_len,
            )
        )
        if sequence_segmentations is None:
            if not os.path.isfile(cache_pkl_path):
                self.build_segmentation_cache()
                assert(os.path.isfile(cache_pkl_path))

            with open(cache_pkl_path, 'rb') as f:
                sequence_segmentations = pickle.load(f)

        # Register entries
        logger.info('Initialized dataset class for: %s' % self.path)

    def build_segmentation_cache(self):
        """Create support data structure for knowing how to segment (cut up) time sequences."""
        all_folders = sorted([
            d for d in os.listdir(self.path) if os.path.isdir(self.path + '/' + d)
        ])
        output_to_cache = {}
        for folder_name in all_folders:
            participant_path = '%s/%s' % (self.path, folder_name)
            assert(os.path.isdir(participant_path))
            output_to_cache[folder_name] = {}

            subfolders = sorted([
                p for p in os.listdir(participant_path)
                if os.path.isdir(os.path.join(participant_path, p))
                and p.split('/')[-1].startswith('step')
                and 'eye_tracker_calibration' not in p
            ])
            for subfolder in subfolders:
                subfolder_path = '%s/%s' % (participant_path, subfolder)
                output_to_cache[folder_name][subfolder] = {}

                # NOTE: We assume that the videos are synchronized and have the same length in time.
                #       This should be the case for the publicly released EVE dataset.
                for source in ('screen', 'basler', 'webcam_l', 'webcam_c', 'webcam_r'):
                    current_outputs = []
                    source_path_pre = '%s/%s' % (subfolder_path, source)
                    available_indices = np.loadtxt('%s.timestamps.txt' % source_path_pre)
                    num_available_indices = len(available_indices)

                    # Determine desired length and skips
                    fps = source_to_fps[source]
                    target_len_in_s = eve_config.max_sequence_len / eve_config.assumed_frame_rate
                    num_original_indices_in_sequence = fps * target_len_in_s
                    assert(num_original_indices_in_sequence.is_integer())
                    num_original_indices_in_sequence = int(
                        num_original_indices_in_sequence
                    )
                    index_interval = int(fps / eve_config.assumed_frame_rate)
                    start_index = 0
                    while start_index < num_available_indices:
                        end_index = min(
                            start_index + num_original_indices_in_sequence,
                            num_available_indices
                        )
                        picked_indices = list(range(start_index, end_index, index_interval))
                        current_outputs.append(picked_indices)

                        # Move along sequence
                        start_index += num_original_indices_in_sequence

                    # Store back indices
                    if len(current_outputs) > 0:
                        output_to_cache[folder_name][subfolder][source] = current_outputs
                        # print('%s: %d' % (source_path_pre, len(current_outputs)))

        # Do the caching
        with open(cache_pkl_path, 'wb') as f:
            pickle.dump(output_to_cache, f)

        logger.info('> Stored indices of sequences to: %s' % cache_pkl_path)

    def preprocess_frames(self, frames):
        # Expected input:  N x H x W x C
        # Expected output: N x C x H x W
        frames = np.transpose(frames, [0, 3, 1, 2])
        frames = frames.astype(np.float32)
        frames *= 2.0 / 255.0
        frames -= 1.0
        return frames

    def preprocess_frames_raw(self, frames):
        # Expected input:  N x H x W x C
        # Expected output: N x C x H x W
        frames = np.transpose(frames, [0, 3, 1, 2])
        frames = frames.astype(np.float32)
        # frames *= 2.0 / 255.0
        # frames -= 1.0
        return frames

    def preprocess_screen_frames(self, frames):
        # Expected input:  N x H x W x C
        # Expected output: N x C x H x W
        frames = np.transpose(frames, [0, 3, 1, 2])
        frames = frames.astype(np.float32)
        frames *= 1.0 / 255.0
        return frames

    screen_frames_cache = {}

    def load_all_from_source(self, path, source, selected_indices, frame_type='eyes'):
        assert(source in ('basler', 'webcam_l', 'webcam_c', 'webcam_r', 'screen'))

        # Read HDF
        subentry = {}  # to output
        if source != 'screen':
            with h5py.File('%s/%s.h5' % (path, source), 'r') as hdf:
                for k1, v1 in hdf.items():
                    if isinstance(v1, h5py.Group):
                        subentry[k1] = np.copy(v1['data'][selected_indices])
                        subentry[k1 + '_validity'] = np.copy(v1['validity'][selected_indices])
                    else:
                        shape = v1.shape
                        subentry[k1] = np.repeat(np.reshape(v1, (1, *shape)),
                                                 repeats=eve_config.max_sequence_len, axis=0)

            # Compute rotation matrices from rvec values
            subentry['head_R'] = np.stack([cv.Rodrigues(rvec)[0] for rvec in subentry['head_rvec']])

        if eve_config.load_full_frame_for_visualization and source == 'screen':
            _, full_frames = VideoReader(path + '/' + source + '.mp4',
                                         frame_indices=selected_indices,
                                         is_async=False,
                                         video_decoder_codec=eve_config.video_decoder_codec).get_frames()
            subentry['full_frame'] = full_frames

        # Get frames
        video_path = '%s/%s' % (path, source)
        output_size = None
        if source == 'screen':
            video_path += '.128x72.mp4'
            output_size = eve_config.screen_size

        else:
            if frame_type == 'full':
                video_path += '.mp4'
            elif frame_type == 'face':
                video_path += '_face.mp4'
                output_size = (eve_config.face_size[0], eve_config.face_size[1])
            elif frame_type == 'eyes':
                video_path += '_eyes.mp4'
                output_size = (2*eve_config.eyes_size[0], eve_config.eyes_size[1])
            else:
                raise ValueError('Unknown camera frame type: %s' % eve_config.camera_frame_type)

        timestamps, frames = VideoReader(video_path, frame_indices=selected_indices,
                                             is_async=False, output_size=output_size, 
                                             video_decoder_codec=eve_config.video_decoder_codec).get_frames()

        # Collect and return
        subentry['timestamps'] = np.asarray(timestamps, dtype=int)
        frames = (
            self.preprocess_screen_frames(frames)
            if source == 'screen' else
            self.preprocess_frames_raw(frames)
        )
        if source == 'screen':
            subentry['frame'] = frames
        else:
            if frame_type == 'face':
                subentry['face_patch'] = frames
            else:
                ew, eh = eve_config.eyes_size
                subentry['right_patch'] = frames[:, :, :, ew:]
                subentry['left_patch'] = frames[:, :, :, :ew]

        return subentry

    def process_all_stimuli(self, stimuli_dir, camera, full_path, index_list, stimulus_subindices):
        all_stimuli = {}
        all_stimuli['eyes'] = self.load_all_from_source(full_path, camera, index_list, 'eyes')
        all_stimuli['face'] = self.load_all_from_source(full_path, camera, index_list, 'face')
        for stimulus_number, stimulus_subindex in stimulus_subindices.items():
            for patch_type in ['left', 'right', 'face']:
                for time in range(len(stimulus_subindex)):
                    entry_dir = os.path.join(stimuli_dir, camera, str(stimulus_number), patch_type)
                    entry_path = os.path.join(entry_dir, str(time) + '.pbz2')

                    # Get frame
                    entry = {}
                    src_stimuli = all_stimuli['face'] if patch_type == 'face' else all_stimuli['eyes']
                    entry['frame'] = src_stimuli[patch_type + '_patch'][stimulus_subindex[time]]

                    # Fetch ground truth
                    get_values = {'cam_gaze_dir': patch_type + '_g_tobii', 'head_dir': 'face_h', 'gaze_pos': patch_type + '_o', 'head_pos': 'face_o'}
                    for tar_key, src_key in get_values.items():
                        if src_key in src_stimuli:
                            entry[tar_key] = src_stimuli[src_key][stimulus_subindex[time]]
                    
                    # Convert to relative gaze
                    if 'cam_gaze_dir' in entry:
                        entry['gaze_dir'] = entry['cam_gaze_dir'] - entry['head_dir']
                    
                    # Convert directions to vector form
                    for key in entry.keys():
                        if '_dir' in key:
                            entry[key] = pitch_yaw_to_vector(entry[key])

                    with bz2.BZ2File(entry_path, 'w') as f:
                        cPickle.dump(entry, f)
        return "Processed {:s}, Camera {:s}".format(full_path, camera)


    def preprocess(self, output_dir):
        patches = MultiDict(['sub', 'head', 'gaze', 'app'])
        num_processes = 32
        if num_processes > 0:
            pool = Pool(processes=num_processes)

        for participant_name, participant_data in sequence_segmentations.items():
            if participant_name not in self.participants_to_use:
                continue

            for stimulus_name, stimulus_segments in participant_data.items():
                current_stimulus_type = stimulus_type_from_folder_name(stimulus_name)
                if current_stimulus_type not in self.types_of_stimuli:
                    continue
                if len(self.stimulus_name_includes) > 0:
                    if self.stimulus_name_includes not in stimulus_name:
                        continue

                for camera, all_indices in stimulus_segments.items():
                    if camera not in self.cameras_to_use:
                        continue

                    index_list = []
                    stimulus_subindices = {}
                    index_offset = 0
                    for stimulus_number, indices in enumerate(all_indices):
                        stimulus_subindices[stimulus_number] = np.array(list(range(index_offset, index_offset+len(indices))), dtype=int)
                        index_list.extend(indices)
                        index_offset += len(indices)

                    full_path = '%s/%s/%s' % (self.path, participant_name, stimulus_name)
                    stimuli_dir_rel = os.path.join(participant_name, stimulus_name)
                    stimuli_dir = os.path.join(output_dir, stimuli_dir_rel)

                    for stimulus_number, stimulus_subindex in stimulus_subindices.items():
                        for patch_type in ['left', 'right', 'face']:
                            for time in range(len(stimulus_subindex)):
                                entry_dir = os.path.join(stimuli_dir, camera, str(stimulus_number), patch_type)
                                os.makedirs(entry_dir, exist_ok=True)
                                entry_dir_rel = os.path.join(stimuli_dir_rel, camera, str(stimulus_number), patch_type)
                                entry_path_rel = os.path.join(entry_dir_rel, str(time) + '.pbz2')

                                sub = frozendict({'participant': participant_name, 'stimulus_name': stimulus_name, 'stimulus_number': stimulus_number})
                                tags = {'sub': sub, 'head': camera, 'app': patch_type, 'gaze': time}
                                patches[tags] = entry_path_rel
                    
                    if num_processes == 0:
                        print(self.process_all_stimuli(stimuli_dir, camera, full_path, index_list, stimulus_subindices))
                    else:
                        pool.apply_async(self.process_all_stimuli, (stimuli_dir, camera, full_path, index_list, stimulus_subindices), callback=print)
            
        # Wait for all processes to finish
        if num_processes > 0:
            pool.close()
            pool.join()
            print("All processes complete")

        index_path = os.path.join(output_dir, 'index.pbz2')
        with bz2.BZ2File(index_path, 'w') as f:
            cPickle.dump(patches, f)
        print("Finished preprocessing eve")


eve_input_path = '/home/ubuntu/data/eve_dataset/'
eve_output_path = '/home/ubuntu/data/eve_preprocessed/'
eve_cameras = ['basler', 'webcam_l', 'webcam_c', 'webcam_r']
eve_stimuli = ['image', 'video', 'wikipedia']

if __name__ == '__main__':
    for fold_name in ['train', 'val', 'test']:
        dataset = EveDataset(config.eve_raw_path,
                             participants_to_use=predefined_eve_splits[fold_name],
                             cameras_to_use=eve_cameras,
                             types_of_stimuli=eve_stimuli)
        dataset.preprocess(os.path.join(config.eve_preprocessed_path, fold_name))
