import tensorflow as tf
from fjcommon import tf_helpers
import numpy as np

from data import Data
from utils import Utils
from distortions import Distortions
from network import Autoencoder, Probability_Model

from config import directories, config_model

class Model(object):
    def __init__(self, config, statistics, input_shape, batch_size, epoch_size, dataset_test, dataset_train=None, evaluate=False):
        self.G_global_step = tf.Variable(0, trainable=False)
        self.D_global_step = tf.Variable(0, trainable=False)

        self.handle = tf.placeholder(tf.string, shape=[])
        self.training_phase = tf.placeholder(tf.bool)

        self.mean, self.std, _ = Utils.get_statistics(statistics)
        self.input_shape = input_shape
        self.epoch_size = epoch_size

        self.config = config

        self.autoencoder = Autoencoder(self.config, self.mean, self.std)
        self.probability_model = Probability_Model(config_model.num_centers)

        self.test_path_placeholder = tf.placeholder(dataset_test.dtype)
        test_dataset = Data.load_dataset(self.test_path_placeholder, batch_size, training=False)

        self.iterator = tf.data.Iterator.from_string_handle(self.handle,
                                                            test_dataset.output_types,
                                                            self.input_shape)

        self.test_iterator = test_dataset.make_initializable_iterator()

        if dataset_train is not None:
            self.train_path_placeholder = tf.placeholder(dataset_train.dtype, dataset_train.shape)
            train_dataset = Data.load_dataset(self.train_path_placeholder, batch_size, training=True)
            self.train_iterator = train_dataset.make_initializable_iterator()

        self.x = self.iterator.get_next()

        self.z_masked, self.t = self.autoencoder.encoder(self.x, training=self.training_phase)

        self.z_hat, self.symbols = self.autoencoder.quantizer(self.z_masked, training=self.training_phase)

        self.reconstruction = self.autoencoder.generator(self.z_hat, training=self.training_phase)

        self.z_hat = tf.stop_gradient(self.z_hat)

        self.bitcost_value = self.probability_model.bitcost(self.z_hat, self.symbols, training=self.training_phase, pad_value=self.autoencoder.centers[0])
        self.bpv = Utils.bitcost_to_bpv(self.x, self.bitcost_value)

        self.distortion = Distortions(self.x, self.reconstruction)

        if evaluate:
            return

        if np.random.random_sample() > 0.5:
            self.D_x  = self.autoencoder.discriminator(self.x, training=self.training_phase, reuse=False)
            self.D_Gz = self.autoencoder.discriminator(self.reconstruction, training=self.training_phase, reuse=True)
        else:
            self.D_Gz = self.autoencoder.discriminator(self.reconstruction, training=self.training_phase, reuse=False)
            self.D_x  = self.autoencoder.discriminator(self.x, training=self.training_phase, reuse=True)

        # loss ---
        self.D_loss = tf.reduce_mean(tf.square(self.D_x - 1.)) + tf.reduce_mean(tf.square(self.D_Gz))
        self.G_loss = tf.reduce_mean(tf.square(self.D_Gz - 1.))

        self.loss, self.H_real, self.pm_comps, self.ae_comps = self.get_loss(self.t, self.bitcost_value, self.distortion)

        self.G_loss += self.loss

        self.fetch_dict = {
            'G_loss':   self.G_loss,
            'D_loss':   self.D_loss,
            'psnr':     self.distortion.psnr,
            'mse':      self.distortion.mse,
            'bpv':      self.bpv,
            'H_real':   self.H_real,
            'centers':  self.autoencoder.centers
        }

        self.G_train_op, self.D_train_op = self.get_train_op(self.G_loss, self.D_loss)


    def get_loss(self, t, bitcost_value, distortion):
        bc_mask = (bitcost_value * t)
        
        H_real = tf.reduce_mean(bitcost_value, name='H_real')
        H_mask = tf.reduce_mean(bc_mask, name='H_mask')
        H_soft = 0.5 * (H_mask + H_real)

        H_target = tf.constant(self.config.H_target, tf.float32, name='H_target')
        beta = tf.constant(self.config.beta, tf.float32, name='beta')

        pc_loss = beta * tf.maximum(H_soft - H_target, 0)

        reg_probability_model = 0
        reg_encoder = tf.losses.get_regularization_loss(scope='encoder')
        reg_generator = tf.losses.get_regularization_loss(scope='generator')
        reg_loss = reg_probability_model + reg_encoder + reg_generator

        pm_comps = [
            ('H_mask', H_mask),
            ('H_real', H_real),
            ('pc_loss', pc_loss),
            ('reg', reg_probability_model)
        ]

        ae_comps = [
            ('d_loss', distortion.d_loss),
            ('reg_enc_dec', reg_encoder+reg_generator)
        ]

        loss = pc_loss + reg_loss + distortion.d_loss

        return loss, H_real, pm_comps, ae_comps


    def get_train_op(self, G_loss_train, D_loss_train):
        with tf.name_scope('lr_ae'):
            learning_rate = tf.constant(8e-5, tf.float32, name='lr_initial')
            lr_ae = tf.train.exponential_decay(
                learning_rate,
                self.G_global_step,
                decay_steps=self.epoch_size * 10,
                decay_rate=0.05,
                staircase=True
            )

        with tf.name_scope('lr_pm'):
            learning_rate = tf.constant(8e-5, tf.float32, name='lr_initial')
            lr_pm = tf.train.exponential_decay(
                learning_rate,
                self.G_global_step,
                decay_steps=self.epoch_size * 10,
                decay_rate=0.05,
                staircase=True
            )

        optimizer_ae = tf.train.AdamOptimizer(learning_rate=lr_ae, name='Adam_AE')
        optimizer_pc = tf.train.AdamOptimizer(learning_rate=lr_pm, name='Adam_PC')
        optimizer_d  = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5)

        vars_probability_model = Utils.scope_variables('probability_model')
        vars_disciminator = Utils.scope_variables('discriminator')

        prob_optimizer_and_vars = [(optimizer_pc, vars_probability_model)]

        G_train_op = tf_helpers.create_train_op_with_different_lrs(G_loss_train, optimizer_ae, prob_optimizer_and_vars, summarize_gradients=False)

        D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')
        
        # Execute the update_ops before performing the train_step
        with tf.control_dependencies(D_update_ops):
            D_opt_op = optimizer_d.minimize(D_loss_train, name='optimizer_d', global_step=self.D_global_step, var_list=vars_disciminator)

        D_ema = tf.train.ExponentialMovingAverage(decay=0.999, num_updates=self.D_global_step)
        D_maintain_averages_op = D_ema.apply(vars_disciminator)

        with tf.control_dependencies(D_update_ops+[D_opt_op]):
            D_train_op = tf.group(D_maintain_averages_op)

        return G_train_op, D_train_op