import os
import tensorflow as tf
from tensorflow import keras
from config import test_conf, train_conf, conf
from utils import make_split,  make_work_dir


def main():
    # make_work_dir()
    # make_split()
    loaded_model = keras.models.load_model(test_conf.latest_best)
    load_dataset = keras.utils.image_dataset_from_directory(conf.test_dir,image_size=train_conf.img_shape, batch_size=train_conf.batch_size)


main()