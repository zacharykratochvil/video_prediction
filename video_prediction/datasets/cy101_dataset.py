import os
from .base_dataset import VideoDataset


class CY101VideoDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        super(CY101VideoDataset, self).__init__(*args, **kwargs)
        self.dataset_name = os.path.basename(os.path.split(self.input_dir)[0])
        self.state_like_names_and_shapes['images'] = 'image_%d', (64, 64, 3)
        self._check_or_infer_shapes()

    def get_default_hparams_dict(self):
        pass

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