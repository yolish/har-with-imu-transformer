from skimage.io import imread
from torch.utils.data import Dataset
import pandas as pd
from os.path import join
import numpy as np
import logging


class IMUDataset(Dataset):
    """
        A class representing a dataset for IMU learning tasks
    """
    def __init__(self, imu_dataset_file, window_size, input_size,
                 window_shift=None):
        """
        :param imu_dataset_file: (str) a file with imu signals and their labels
        :param window_size (int): the window size to consider
        :param input_size (int): the input size (e.g. 6 for 6 IMU measurements)
        :param window_shift (int): the overlap between each window
        :return: an instance of the class
        """
        super(IMUDataset, self).__init__()
        if window_shift is None:
            window_shift = window_size
        df = pd.read_csv(imu_dataset_file)
        if df.shape[1] == 1:
            df = pd.read_csv(imu_dataset_file, delimiter='\t')
        # Fetch the flatten IMU data and labels
        self.imu = df.iloc[:, :input_size].values
        self.labels = df.iloc[:, input_size:].values
        n = self.labels.shape[0]
        self.start_indices = list(range(0, n - window_size + 1, window_shift))
        self.window_size = window_size
        logging.info(
            "Number of windows: {} (generated from {} windows of size {} with shift {})".format(len(self.start_indices),
                                                                                                n // window_size,
                                                                                                window_size,
                                                                                                window_shift))

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start_index = self.start_indices[idx]
        window_indices = list(range(start_index, (start_index + self.window_size)))
        imu = self.imu[window_indices, :]
        window_labels = self.labels[window_indices, :]
        if len(np.unique(window_labels)) > 1:
            logging.warning("Window includes more than one class present, introducing noise")
        label = window_labels[0][0]
        sample = {'imu': imu,
                  'label': label}
        return sample



