import argparse
import glob
import itertools
import os
import re

import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image

from video_prediction.datasets.base_dataset import VarLenFeatureVideoDataset
from video_prediction.datasets import cy101_metadata as cy_meta


class CY101VideoDataset(VarLenFeatureVideoDataset):
    def __init__(self, *args, **kwargs):
        super(CY101VideoDataset, self).__init__(*args, **kwargs)
        from google.protobuf.json_format import MessageToDict
        example = next(tf.python_io.tf_record_iterator(self.filenames[0]))
        dict_message = MessageToDict(tf.train.Example.FromString(example))
        feature = dict_message['features']['feature']
        image_shape = tuple(int(feature[key]['int64List']['value'][0]) for key in ['height', 'width', 'channels'])
        self.state_like_names_and_shapes['images'] = 'images/encoded', image_shape


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
        with open(os.path.join(self.input_dir, 'sequence_lengths.txt'), 'r') as sequence_lengths_file:
            sequence_lengths = sequence_lengths_file.readlines()
        sequence_lengths = [int(sequence_length.strip()) for sequence_length in sequence_lengths]
        return np.sum(np.array(sequence_lengths) >= self.hparams.sequence_length)


    @property
    def jpeg_encoding(self):
        return True


##############
# functions from kth_dataset.py for saving tfrecords
##############
def _bytes_list_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#####
# takes an output file name, list of sequences of images, and list of actions
# saves as tfrecord file
#####
def save_tf_record(output_fname, sequences, actions):
    print('saving sequences to %s' % output_fname)
    with tf.python_io.TFRecordWriter(output_fname) as writer:
        for sequence, action in zip(sequences, actions):
            num_frames = len(sequence)
            height, width, channels = sequence[0].shape
            encoded_sequence = [bytes(image) for image in sequence]
            encoded_action = [bytes(action.tostring())]
            features = tf.train.Features(feature={
                'sequence_length': _int64_feature(num_frames),
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'channels': _int64_feature(channels),
                'images/encoded': _bytes_list_feature(encoded_sequence),
                'action': _bytes_list_feature(encoded_action)
            })
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())


##############
# creates and returns a pandas dataframe of directories and associated metadata
#   columns: full_path, object, trial, exec, behavior
#   rows: one for every folder in the dataset
##############
def get_metadata(input_dir):

    ## locate files and validate directory structure
    all_files = glob.glob(os.sep.join([input_dir, 'vision*', "*", "*", "*", "*"]))
    if len(all_files) == 0:
        raise Exception(f"{input_dir} is not a properly formatted raw data " +
                        "directory for the CY101 dataset.")
    
    ## parse metadata from folder names
    file_df = pd.DataFrame(all_files, columns=["full_path"])
    path_df = file_df.copy()

    folder_meanings = ["object", "trial", "exec", "behavior"]
    num_meanings = len(folder_meanings)
    split_df = file_df.apply(lambda x : pd.Series(x.tolist()[0].split(os.sep)[-num_meanings:], index=folder_meanings), axis=1)
    
    file_df = pd.concat([path_df, split_df], axis=1)

    return file_df


# selects relevant files and apportions them into train and test folders
def partition_data(xval_test_objects, file_metadata, chosen_behaviors):
    chosen_behaviors = set(chosen_behaviors)

    all_splits = []
    drops = []
    for file_i, file_data in file_metadata.iterrows():
        
        split_folders = []
        for test_objects in xval_test_objects:
        
            # validate object, behavior, then do split
            if file_data["object"] not in cy_meta.OBJECTS:
                drops.append(file_i)
                continue
            elif file_data["behavior"] not in chosen_behaviors:
                drops.append(file_i)
                continue
            elif file_data["object"] in test_objects:
                split_folders.append("test")
            else:
                split_folders.append("train")

        all_splits.append(split_folders)

    # prepare new paths for merge with metadata
    splits_df = pd.DataFrame(all_splits)
    num_folds = len(xval_test_objects)
    splits_df = splits_df.apply(lambda x: pd.Series(x.tolist(), index=range(num_folds)), axis=1)

    # update and return the file_metadata dataframe
    file_metadata.drop(drops, axis=0, inplace=True)
    return pd.concat([file_metadata, splits_df], axis=1)


# crop sequence length and resize images
def generate_npy_vision(path, behavior, img_size):
    """
    :param path: path to images folder,
    :return: numpy array with size [SUB_SAMPLE_SIZE, SEQ_LENGTH, ...]
    """
    # manually crops image sequence to where there is interesting data
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

    files = sorted(glob.glob(os.path.join(path, '*.jpg')))
    files = files[crop_stategy[behavior][0]:crop_stategy[behavior][1]]
    img_list = []
    for file in files:
        img = Image.open(file)
        img = img.resize([img_size,img_size])
        img = np.array(img)
        img_list.append(img)

    return img_list


# create vector encoding descriptors of an object
def switch_words_on(object, descriptor_codes, descriptors_by_object):
    encoded_output = np.zeros(len(descriptor_codes))
    if type(object) != type(None):
        for descriptor in descriptors_by_object[object]:
            word_index = descriptor_codes[descriptor]
            encoded_output[word_index] = 1
    return encoded_output


def compute_behavior(behavior, object):
    out_behavior_npys = np.zeros(len(cy_meta.BEHAVIORS))
    out_behavior_npys[cy_meta.BEHAVIORS.index(behavior)] = 1
    descriptors = switch_words_on(object, cy_meta.DESCRIPTOR_CODES, cy_meta.DESCRIPTORS_BY_OBJECT)
    out_behavior_npys = np.hstack([out_behavior_npys, descriptors])
    return out_behavior_npys

####
# copies relevant data from directories in partitioned_metadata
# to the partition_dir, resizing images to image_size
# and puting sequences_per_file in each tfrecords file
####
def copy_data(partitioned_metadata, partition_dir, image_size, sequences_per_file=128):
    
    sequence_lengths_file = open(os.path.join(partition_dir, 'sequence_lengths.txt'), 'w')
    sequences = []
    actions = []
    for sequence_i, row_i_and_row in enumerate(partitioned_metadata.iterrows()):
        row = row_i_and_row[1]

        # preprocess and store images and their properties
        frames = generate_npy_vision(str(row["full_path"]), str(row["behavior"]), image_size)
        sequence_lengths_file.write("%d\n" % len(frames))
        sequences.append(frames)

        # preprocess and store behavior data associated with each sequence of images
        actions.append(compute_behavior(row["behavior"], row["object"]))

        # actually write to file in batches
        if ((sequence_i + 1) % sequences_per_file) == 0 or sequence_i >= (partitioned_metadata.shape[0] - 1):
            num_sequences_to_save = (sequence_i % sequences_per_file) + 1 # all numbers mapped to range [1 sequences_per_file]
            output_fname = 'sequence_{0}_to_{1}.tfrecords'.format((sequence_i+1)-num_sequences_to_save, sequence_i)
            output_fname = os.path.join(partition_dir, output_fname)
            save_tf_record(output_fname, sequences, actions)
            sequences[:] = []
            actions[:] = []
    
    sequence_lengths_file.close()


# processes raw data into tfrecords based on command-line specified args
def main(args):
    # process and validate behavior argument
    if len(args.behavior) == 0:
        args.behavior = cy_meta.BEHAVIORS
    else:
        for name in args.behavior:
            if name not in cy_meta.BEHAVIORS:
                raise Exception(f"{name} not a behavior in the cy101 dataset. " +
                                f"Available behaviors are {cy_meta.BEHAVIORS}")

    # validate data folder and parse metadata
    print("\nparsing input data directory structure...")
    file_metadata = get_metadata(args.input_dir)
    print("done. preview:")
    print(file_metadata.head())
    
    # 5-fold split to prepare for x-val and remove undesired behaviors
    print("\npartitioning files for cross validation...")
    xval_test_objects = cy_meta.category_split(args.num_folds)
    partitioned_metadata = partition_data(xval_test_objects, file_metadata, args.behavior)
    print("done. preview:")
    print(partitioned_metadata.head())

    # compile data into tfrecords
    print("\ncopying files to output directory...")
    for xval_i in range(args.num_folds):
        for folder in ["train", "test"]:
            
            partition_dir = os.path.join(args.output_dir, str(xval_i), folder)
            if not os.path.exists(partition_dir):
                os.makedirs(partition_dir)
            
            this_partition_only = partitioned_metadata[partitioned_metadata[xval_i]==folder]
            copy_data(this_partition_only, partition_dir, args.image_size)
    

if __name__ == '__main__':
    # specify and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="directory containing the processed directories "
                                                    "boxing, handclapping, handwaving, "
                                                    "jogging, running, walking")
    parser.add_argument("output_dir", type=str)
    parser.add_argument("image_size", type=int)
    parser.add_argument('--behavior', nargs="+", action="append", default=[], help='which behavior?')
    parser.add_argument('--num_folds', type=int, default=5, help="Number of folds for k-fold cross validation.")
    args = parser.parse_args()

    # process data into tfrecords
    main(args)