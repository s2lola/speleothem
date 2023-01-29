import numpy as np
from PIL import Image
import tensorflow as tf

class Speleothem_dataset():
    def __init__(self, speleothem_file: str) -> None:
        self.speleothem_image = Image.open(speleothem_file).convert("L")
        self.speleothem_df = np.array(self.speleothem_image)

        n = len(self.speleothem_df)
        self.train_df = self.speleothem_df[0:int(n*0.7)]
        self.validation_df = self.speleothem_df[int(n*0.7):int(n*0.9)]
        self.test_df = self.speleothem_df[int(n*0.9):]

        train_mean = self.train_df.mean()
        train_std = self.train_df.std()

        self.train_df = self.normalization(self.train_df, train_mean, train_std)
        self.validation_df = self.normalization(self.validation_df, train_mean, train_std)
        self.test_df = self.normalization(self.validation_df, train_mean, train_std)
    
    def normalization(self, df, mean, std):
        return (df - mean) / std


class WindowGenerator():
    def __init__(self, input_width: int, label_width: int, shift: int,
                 train_df, validation_df, test_df) -> None:
        self.train_df = train_df
        self.validation_df = validation_df
        self.test_df = test_df

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    @property
    def train(self):
        return self.make_dataset(self.train_df)
    
    @property
    def validation(self):
        return self.make_dataset(self.validation_df)
    
    @property
    def test(self):
        return self.make_dataset(self.test_df)

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        
        return inputs, labels
    
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,
        )

        ds = ds.map(self.split_window)

        return ds