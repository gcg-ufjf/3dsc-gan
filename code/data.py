import os
import tensorflow as tf
import numpy as np

class Data(object):

    @staticmethod
    def load_dataframe(file_name):
        list_datasets  = []

        datasets = list()
        with open(file_name) as file:
            for path in file:
                if path.strip():
                    path = path.replace('\n','')

                    list_datasets.append(path)

                    files = os.listdir(path)
                    for f in files:
                        if f.endswith('.npy') and not 'slice' in f:
                            datasets.append(os.path.join(path, f))

        return list_datasets, np.array(datasets)

    @staticmethod
    def load_dataset(image_paths, batch_size, training=True):
        def _parser(image_path):
            def _image_load(path):
                file = np.load(path)
                file = np.expand_dims(file, -1)
                file = np.transpose(file, (3, 2, 0, 1))
                return file

            image = tf.py_func(_image_load, [image_path], tf.float32, stateful=False, name='img_load')

            return image

        dataset = tf.data.Dataset.from_tensor_slices(image_paths)

        if training:
            dataset_size = image_paths.get_shape().as_list()[0]
            dataset = dataset.shuffle(buffer_size=dataset_size, reshuffle_each_iteration=True)

        dataset = dataset.map(_parser)
        #dataset = dataset.batch(batch_size, drop_remainder=True) # Tensorflow >= 1.10
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

        if not training:
            dataset = dataset.repeat()

        return dataset

    @staticmethod
    def load_inference(datasets_path, dataset_name, dir):
        import glob

        dataset_path = os.path.join(datasets_path, dataset_name) + '/'
        list_files = glob.glob(dataset_path + "**/*"+dir+"_vol_slice_*_ori.npy", recursive=True)
        list_files = sorted(list_files, key=os.path.getmtime)

        return np.array(list_files)