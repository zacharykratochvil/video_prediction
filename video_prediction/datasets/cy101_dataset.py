import argparse
import glob
import itertools
import os
import pickle
import random
import re

import numpy as np
import skimage.io
import tensorflow as tf

from .base_dataset import VideoDataset


class CY101VideoDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        super(CY101VideoDataset, self).__init__(*args, **kwargs)
        self.dataset_name = os.path.basename(os.path.split(self.input_dir)[0])
        self.state_like_names_and_shapes['images'] = 'image_%d', (64, 64, 3)
        self._check_or_infer_shapes()

    def get_default_hparams_dict(self):
        default_hparams = super(CY101VideoDataset, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=4,
            sequence_length=20,
            force_time_shift=True,
            shuffle_on_val=True,
            use_state=False,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    def num_examples_per_epoch(self):
        # extract information from filename to count the number of trajectories in the dataset
        count = 0
        for filename in self.filenames:
            match = re.search('traj_(\d+)_to_(\d+).tfrecords', os.path.basename(filename))
            start_traj_iter = int(match.group(1))
            end_traj_iter = int(match.group(2))
            count += end_traj_iter - start_traj_iter + 1

        # alternatively, the dataset size can be determined like this, but it's very slow
        # count = sum(sum(1 for _ in tf.python_io.tf_record_iterator(filename)) for filename in filenames)
        return count

    @property
    def jpeg_encoding(self):
        return True

    crop_stategy = {
    'crush': [16, -5],
    'grasp': [0, -10],
    'lift_slow': [0, -3],
    'shake': [0, -1],
    'poke': [2, -5],
    'push': [2, -5],
    'tap': [0, -5],
    'low_drop': [0, -1],
    'hold': [0, -1],
}


SEQUENCE_LENGTH = 10
STEP = 4
IMG_SIZE = (64, 64)

STRATEGY = 'object' # object | category | trial

        def read_dir(DATA_DIR):
    visions = glob.glob(os.path.join(DATA_DIR, 'vision*/*/*/*/*'))
    return visions


def convert_audio_to_image(audio_path):
    ims, duration = plotstft(audio_path)
    return ims, duration


def generate_npy_vision(path, behavior, sequence_length):
    """
    :param path: path to images folder,
    :return: numpy array with size [SUB_SAMPLE_SIZE, SEQ_LENGTH, ...]
    """
    files = sorted(glob.glob(os.path.join(path, '*.jpg')))
    img_length = len(files)
    files = files[crop_stategy[behavior][0]:crop_stategy[behavior][1]]
    imglist = []
    for file in files:
        img = PIL.Image.open(file)
        img = img.resize(IMG_SIZE)
        img = np.array(img).transpose([2, 0, 1])[np.newaxis, ...]
        imglist.append(img)
    ret = []
    for i in range(0, len(imglist) - sequence_length, STEP):
        ret.append(np.concatenate(imglist[i:i + sequence_length], axis=0))
    return ret, img_length


def generate_npy_haptic(path1, path2, n_frames, behavior, sequence_length):
    """
    :param path: path to ttrq0.txt, you need to open it before you process
    :param n_frames: # frames
    :return: list of numpy array with size [SEQ_LENGTH, ...]
    :preprocess protocol: 48 bins for each single frame, given one frame, if #bin is less than 48,
                            we pad it in the tail with the last bin value. if #bin is more than 48, we take bin[:48]
    """
    if not os.path.exists(path1):
        return None, None
    haplist1 = open(path1, 'r').readlines()
    haplist2 = open(path2, 'r').readlines()
    haplist = [list(map(float, v.strip().split('\t'))) + list(map(float, w.strip().split('\t')))[1:] for v, w in
               zip(haplist1, haplist2)]
    haplist = np.array(haplist)
    time_duration = (haplist[-1][0] - haplist[0][0]) / n_frames
    bins = np.arange(haplist[0][0], haplist[-1][0], time_duration)
    end_time = haplist[-1][0]
    groups = np.digitize(haplist[:, 0], bins, right=False)

    haplist = [haplist[np.where(groups == idx)][..., 1:][:48] for idx in range(1, n_frames + 1)]
    haplist = [np.pad(ht, [[0, 48 - ht.shape[0]], [0, 0]], mode='edge')[np.newaxis, ...] for ht in haplist]
    haplist = haplist[crop_stategy[behavior][0]:crop_stategy[behavior][1]]
    ret = []
    for i in range(0, len(haplist) - sequence_length, STEP):
        ret.append(np.concatenate(haplist[i:i + sequence_length], axis=0).astype(np.float32)[:, np.newaxis, ...])
    return ret, (bins, end_time)


def generate_npy_audio(path, n_frames_vision_image, behavior, sequence_length):
    """
    :param path: path to audio, you need to open it before you process
    :return: list of numpy array with size [SEQ_LENGTH, ...]
    """
    audio_path = glob.glob(path)
    if len(audio_path) == 0:
        return None
    audio_path = audio_path[0]
    img, duration = convert_audio_to_image(audio_path)
 # create a new dimension

    image_height, image_width = img.shape
    image_width = AUDIO_EACH_FRAME_LENGTH * n_frames_vision_image
    img = PIL.Image.fromarray(img)
    img = img.resize((image_height, image_width))

    img = np.array(img)
    img = img[np.newaxis, ...]
    imglist = []
    for i in range(0, n_frames_vision_image):
        imglist.append(img[:, i * AUDIO_EACH_FRAME_LENGTH:(i + 1) * AUDIO_EACH_FRAME_LENGTH, :])
    imglist = imglist[crop_stategy[behavior][0]:crop_stategy[behavior][1]]
    ret = []
    for i in range(0, len(imglist) - sequence_length, STEP):
        ret.append(np.concatenate(imglist[i:i + sequence_length], axis=0).astype(np.float32)[:, np.newaxis, ...])
    return ret


def generate_npy_vibro(path, n_frames, bins, behavior, sequence_length):
    """
    :param path: path to .tsv, you need to open it before you process
    :return: list of numpy array with size [SEQ_LENGTH, ...]
    """
    path = glob.glob(path)
    if not path and not bins:
        return None
    path = path[0]
    vibro_list = open(path).readlines()
    vibro_list = [list(map(int, vibro.strip().split('\t'))) for vibro in vibro_list]

    vibro_list = np.array(vibro_list)
    vibro_time = vibro_list[:, 0]
    vibro_data = vibro_list[:, 1:]
    bins, end_time = bins
    end_time -= bins[0]
    bins -= bins[0]

    v_h_ratio = vibro_time[-1] / end_time
    bins = bins * v_h_ratio

    groups = np.digitize(vibro_time, bins, right=False)

    vibro_data = [vibro_data[np.where(groups == idx)] for idx in range(1, n_frames + 1)]

    vibro_data = [np.vstack([np.resize(vibro[:, 0], (128,)),
                             np.resize(vibro[:, 1], (128,)),
                             np.resize(vibro[:, 2], (128,))]).T[np.newaxis, ...]
                  for vibro in vibro_data]
    # haplist = [np.pad(ht, [[0, 48 - ht.shape[0]], [0, 0]], mode='edge')[np.newaxis, ...] for ht in haplist]
    vibro_data = vibro_data[crop_stategy[behavior][0]:crop_stategy[behavior][1]]

    ret = []
    for i in range(0, len(vibro_data) - sequence_length, STEP):
        ret.append(np.concatenate(vibro_data[i:i + sequence_length], axis=0).astype(np.float32)[:, np.newaxis, ...])
    return ret


# splits words on objects with balanced categories to prepare for
# 5-fold cross validation
# assumes objects are in groupings/categories of exactly 5 with unique prefixes
def split():
    # test assumptions
    if len(SORTED_OBJECTS) != 100:
        raise Exception("split is intended to work for exactly 100 objects")

    # semi-randomly split data
    splits = [set([]),set([]),set([]),set([]),set([])]
    # for each of 20 categories of objects
    for category_i in range(len(SORTED_OBJECTS)//5):
        low_ind = 5*category_i
        random_list = np.random.permutation(5)

        # for each of the 5 objects in that category
        for object_i in range(5):
            ind = low_ind + random_list[object_i]
            if SORTED_OBJECTS[ind][0] != SORTED_OBJECTS[low_ind][0]:
                raise Exception("each grouping must have exactly 5 objects with identical prefix")
            else:
                splits[object_i].add(SORTED_OBJECTS[ind])

    return splits

# create vector encoding descriptors of an object
def switch_words_on(object, descriptor_codes, descriptors_by_object):
    encoded_output = np.zeros(len(descriptor_codes))
    if type(object) != type(None):
        for descriptor in descriptors_by_object[object]:
            word_index = descriptor_codes[descriptor]
            encoded_output[word_index] = 1
    return encoded_output


def process(visions, chosen_behaviors, OUT_DIR):

    for split_num in range(5):
        train_subdir = 'train'
        test_subdir = 'test'
        # vis_subdir = 'vis'
        if not os.path.exists(os.path.join(OUT_DIR, str(split_num), train_subdir)):
            os.makedirs(os.path.join(OUT_DIR, str(split_num), train_subdir))

        if not os.path.exists(os.path.join(OUT_DIR, str(split_num), test_subdir)):
            os.makedirs(os.path.join(OUT_DIR, str(split_num), test_subdir))

        splits = split()

        fail_count = 0
        for vision in visions:
            print("processing " + vision)

            # The path is object/trial/exec/behavior/file_name (visions does not include file names)
            vision_components = vision.split(os.sep)
            object_name = vision_components[-4]
            behavior_name = vision_components[-1]

            # validate behavior
            if behavior_name not in BEHAVIORS:
                continue      

            # validate object and split
            if object_name not in OBJECTS:
                continue
            if object_name in splits[split_num]:
                subdir = test_subdir
            else:
                subdir = train_subdir
            out_sample_dir = os.path.join(OUT_DIR, str(split_num), subdir, '_'.join(vision.split(os.sep)[-4:]))

            haptic1 = os.path.join(re.sub(r'vision_data_part[1-4]', 'rc_data', vision), 'proprioception', 'ttrq0.txt')
            haptic2 = os.path.join(re.sub(r'vision_data_part[1-4]', 'rc_data', vision), 'proprioception', 'cpos0.txt')
            audio = os.path.join(re.sub(r'vision_data_part[1-4]', 'rc_data', vision), 'hearing', '*.wav')
            vibro = os.path.join(re.sub(r'vision_data_part[1-4]', 'rc_data', vision), 'vibro', '*.tsv')

            out_vision_npys, n_frames = generate_npy_vision(vision, behavior_name, SEQUENCE_LENGTH)
            out_audio_npys = generate_npy_audio(audio, n_frames, behavior_name, SEQUENCE_LENGTH)
            out_haptic_npys, bins = generate_npy_haptic(haptic1, haptic2, n_frames, behavior_name, SEQUENCE_LENGTH)
            out_vibro_npys = generate_npy_vibro(vibro, n_frames, bins, behavior_name, SEQUENCE_LENGTH)

            if out_audio_npys is None or out_haptic_npys is None or out_vibro_npys is None:
                fail_count += 1
                continue
            out_behavior_npys = compute_behavior(chosen_behaviors, behavior_name, object_name)

            for i, (out_vision_npy, out_haptic_npy, out_audio_npy, out_vibro_npy) in enumerate(zip(
                    out_vision_npys, out_haptic_npys, out_audio_npys, out_vibro_npys)):
                ret = {
                    'behavior': out_behavior_npys,
                    'vision': out_vision_npy,
                    'haptic': out_haptic_npy,
                    'audio': out_audio_npy,
                    'vibro': out_vibro_npy
                }
                np.save(out_sample_dir + '_' + str(i), ret)
        print("fail: ", fail_count)

def compute_behavior(CHOSEN_BEHAVIORS, behavior, object):
    out_behavior_npys = np.zeros(len(CHOSEN_BEHAVIORS))
    out_behavior_npys[CHOSEN_BEHAVIORS.index(behavior)] = 1
    descriptors = switch_words_on(object, DESCRIPTOR_CODES, DESCRIPTORS_BY_OBJECT)
    out_behavior_npys = np.hstack([out_behavior_npys, descriptors])
    return out_behavior_npys