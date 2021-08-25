import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import argparse
import time, shutil
import random, math
import numpy as np
import tensorflow as tf

from fjcommon import config_parser

from data import Data
from model import Model
from utils import Utils

from config import directories, config_test, config_model


def validate(config):
    ckpt = tf.train.get_checkpoint_state(directories.checkpoints)

    assert (ckpt.model_checkpoint_path), 'Missing checkpoint file!'

    test_paths = Data.load_inference(directories.datasets_path, config_test.dataset_name, config_test.direction)

    input_shape = Utils.get_input_shape(1, test_paths[0])

    model = Model(config, directories.statistics_file, input_shape, batch_size=1, epoch_size=None,
                  dataset_test=test_paths, evaluate=True)

    saver = tf.train.Saver()

    # create fetch_dict
    fetch_dict = {
        'bpv': model.bpv,
        'psnr': model.distortion.psnr,
        'mse': model.distortion.mse,
        'x_test': model.x,
        'reconstruction': model.reconstruction
    }

    _, _, shape = Utils.get_statistics(directories.statistics_path, config_test.dataset_name)
    
    line_a, line_b, line_c, crop_size = Utils.get_shape_and_crop_size(shape, config_test.direction, test_paths[0])

    ori_cube = np.zeros((line_a, line_b, line_c), dtype=np.float32)

    psnr = snr = -1
    rms_inp = 0
    mse = 0

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        print('\n\n\n------------------------ Testing on '+config_test.dataset_name+' ------------------------\n')

        # Initialize variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        test_handle = sess.run(model.test_iterator.string_handle())

        feed_dict_init = {model.test_path_placeholder: test_paths}
        sess.run(model.test_iterator.initializer, feed_dict=feed_dict_init)

        saver.restore(sess, ckpt.model_checkpoint_path)

        feed_dict = {model.training_phase: False, model.handle: test_handle}
        
        bpv_sum = 0

        slice_count = 0

        for i in range(0, line_a, crop_size[0]):
            for j in range(0, line_b, crop_size[1]):
                for k in range(0, line_c, crop_size[2]):
                    otp = sess.run(fetch_dict, feed_dict=feed_dict)

                    slice_count += 1

                    print('[{}/{}] | PSNR: {:.3f} | BPV: {:.3f} | MSE: {:.6f}'.format(slice_count, len(test_paths), otp['psnr'], otp['bpv'], otp['mse']))

                    max_shape = ori_cube[i:i+crop_size[0], j:j+crop_size[1], k:k+crop_size[2]].shape

                    cube_rec = otp['reconstruction'][0][0]
                    cube_ori = otp['x_test'][0][0]

                    cube_rec = cube_rec[:max_shape[0],:max_shape[1],:max_shape[2]]
                    cube_ori = cube_ori[:max_shape[0],:max_shape[1],:max_shape[2]]

                    rms_inp  += sum((cube_ori**2).ravel())
                    mse      += sum(((cube_ori - cube_rec)**2).ravel())
                    bpv_sum  += otp['bpv']

    print('\n')

    rms_inp /= line_a * line_b * line_c
    rms_inp = math.sqrt(rms_inp)

    mse /= line_a * line_b * line_c

    rms_diff = mse
    rms_diff = math.sqrt(rms_diff)

    psnr = np.float32(10 * np.log10((1.0 - 0.0)**2/ mse))
    snr = np.float32(20 * np.log10(rms_inp / rms_diff))

    bpv = float("{:.6f}".format(bpv_sum / slice_count))

    print('PSNR: {:.3f} | BPV: {:.3f} | MSE: {:.6f}'.format(psnr, bpv, mse))


if __name__ == '__main__':
    validate(config_model)