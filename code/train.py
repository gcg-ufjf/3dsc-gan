import os, sys, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import argparse
import time, shutil
import numpy as np
import tensorflow as tf
from fjcommon import config_parser

from data import Data
from model import Model
from utils import Utils
from create_plots import create_plots

from config import directories, config_model

def train(config):
    list_train_datasets, train_paths = Data.load_dataframe(directories.train_filename)
    list_test_datasets, test_paths = Data.load_dataframe(directories.test_filename)

    test_files = int(len(test_paths) / config.batch_size)
    epoch_size = int(len(train_paths) / config.batch_size)

    input_shape = Utils.get_input_shape(config.batch_size, train_paths[0])

    model = Model(config, directories.statistics_file, input_shape, config.batch_size, epoch_size,
                  test_paths, train_paths, evaluate=False)

    saver = tf.train.Saver(max_to_keep=1)

    feed_dict_train_init = {model.train_path_placeholder: train_paths}
    feed_dict_test_init = {model.test_path_placeholder: test_paths}

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        train_handle = sess.run(model.train_iterator.string_handle())
        test_handle = sess.run(model.test_iterator.string_handle())

        sess.run(model.test_iterator.initializer, feed_dict=feed_dict_test_init)

        best_model = None
        step_time = 0.0

        print('Total steps:', config.epochs * epoch_size)
        print('\n\n\n------------------------ Training ------------------------\n')

        for epoch in range(config.epochs):

            sess.run(model.train_iterator.initializer, feed_dict=feed_dict_train_init)

            while True:
                try:
                    feed_dict = {model.training_phase: True, model.handle: train_handle}

                    otp, step, _, _ = sess.run([model.fetch_dict, model.D_global_step, model.G_train_op, model.D_train_op], feed_dict=feed_dict)

                    print('Epoch {} | Step {} | G_Loss: {:.2f} | D_Loss: {:.3f} | PSNR: {:.2f} | BPV: {:.2f}'.format(epoch, step, otp['G_loss'], otp['D_loss'], otp['psnr'], otp['bpv']))

                    if step % 100 == 0:
                        Utils.write_results(directories.log_dir, step, otp, training=True)

                except tf.errors.OutOfRangeError:
                    # End of epoch
                    print('\nTesting...\n')
                    G_loss = D_loss = mse = psnr = snr = bpv = H_test = 0

                    for _ in range(test_files):
                        feed_dict = {model.training_phase: False, model.handle: test_handle}
                        otp, _ = sess.run([model.fetch_dict, model.D_global_step], feed_dict=feed_dict)

                        G_loss += otp['G_loss'];    D_loss += otp['D_loss'];    mse += otp['mse'];
                        psnr   += otp['psnr'];      bpv += otp['bpv'];          H_test += otp['H_real'];
                        centers = otp['centers'];

                    G_loss /= test_files;  D_loss /= test_files;  mse /= test_files;
                    psnr   /= test_files;  bpv /= test_files;     H_test /= test_files;

                    Utils.write_results(directories.log_dir, step, otp, training=False)
                    Utils.print_results(step, otp, training=False)

                    if not best_model or psnr > best_model:
                        best_model = psnr
                        save_path = saver.save(sess, os.path.join(directories.checkpoints, 'ckpt'), global_step=epoch)

                        print('\nBest model saved to:', save_path, '\n')

                    create_plots(os.path.join(directories.log_dir, 'output.txt'), directories.log_dir)
                    break

                except KeyboardInterrupt:
                    save_path = saver.save(sess, os.path.join(directories.checkpoints, 'ckpt'), global_step=epoch)
                    create_plots(os.path.join(directories.log_dir, 'output.txt'), directories.log_dir)

                    print('Interrupted, model saved to: ', save_path)
                    sys.exit()


if __name__ == '__main__':
    train(config_model)