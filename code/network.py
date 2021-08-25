import tensorflow as tf
from utils import Utils

import numpy as np
from fjcommon import tf_helpers
from tensorflow.contrib import slim as slim

class Autoencoder(object):
    def __init__(self, config, mean, std):
        self.config = config
        self.mean = mean
        self.std = std

        self.reuse_encoder = False
        self.reuse_quantizer = False
        self.reuse_generator = False
        #self.reuse_discriminator = False

        self.centers = Utils.create_centers_variable(self.config)
        Utils.create_centers_regularization_term(self.config, self.centers)


    def encoder(self, x, training):
        print('<------------ Building global seismic generator architecture ------------>')

        def normalize(x):
            return (x - self.mean) / self.std

        def conv_block(x, n_filters, kernel_size=3, stride=2, use_activation=True, **kwargs):
            bn_kwargs = {
                'decay': 0.9,
                'epsilon': 1e-5,
                'scale': True,
                'updates_collections': tf.GraphKeys.UPDATE_OPS,
                'fused': True,
                'is_training': training,
                'data_format': 'NCHW'
            }
            conv_kwargs = {
                'weights_regularizer': slim.l2_regularizer(scale=self.config.regularization_factor),
                'data_format': 'NCDHW',
                'normalizer_fn': slim.batch_norm,
                'normalizer_params': bn_kwargs
            }

            if not use_activation:
                conv_kwargs['activation_fn'] = None

            x = slim.conv3d(x, n_filters, kernel_size, stride, **conv_kwargs)

            return x

        @slim.add_arg_scope
        def residual_block(x, n_outputs, n_conv, **kwargs):
            residual_input = x

            for conv_i in range(n_conv):
                if conv_i == (n_conv - 1):  # no relu after final conv
                    kwargs['use_activation'] = False

                x = conv_block(x, n_outputs, **kwargs)

            return x + residual_input

        def apply_mask(z):
            assert z.shape.ndims == 5, z.shape #NCDHW
            z = z[:,:,0,...] #NCHW

            C = int(z.shape[1]) - 1  # -1 because first channel is heatmap

            t_channel = z[:,0,:,:]  # NHW
            z_without_t = z[:,1:,...]

            t_2D = tf.nn.sigmoid(t_channel) * C  # NHW
            c = tf.range(C, dtype=tf.float32)  # C

            # reshape t_2D and c for broadcasting
            t = tf.expand_dims(t_2D, 1)  # N1HW
            c = tf.reshape(c, (C,1,1))  # C11

            # construct t_3D
            # if heatmap[x, y] == C, then t[x, y, c] == 1 \forall c \in {0, ..., C-1}
            t_3D = tf.maximum(tf.minimum(t-c, 1), 0, name='t_3D')  # NCHW

            z_masked = t_3D * z_without_t

            return z_masked, t_3D
        
        with tf.variable_scope('encoder', reuse=self.reuse_encoder):
            self.reuse_encoder = True

            n_filters = self.config.n_filters

            d = self.config.depth_size
            stride_a = [int(d/4), 2, 2]

            out = normalize(x)
            out = conv_block(out, n_filters // 2, kernel_size=5, stride=2, use_activation=True)
            out = conv_block(out, n_filters ,     kernel_size=5, stride=2, use_activation=True)

            res_kwargs = {
                'weights_regularizer': slim.l2_regularizer(scale=self.config.regularization_factor),
                'data_format': 'NCDHW'
            }
            with slim.arg_scope([residual_block], **res_kwargs):
                residual_input_initial = out
                residual_input_b = out

                out = residual_block(out, n_outputs=n_filters, n_conv=2, kernel_size=3, stride=1)
                out = residual_block(out, n_outputs=n_filters, n_conv=2, kernel_size=3, stride=1)
                out = residual_block(out, n_outputs=n_filters, n_conv=2, kernel_size=3, stride=1)

                out = out + residual_input_b

                out = residual_block(out, n_outputs=n_filters, n_conv=2, kernel_size=3, stride=1, use_activation=False)
                out = out + residual_input_initial
            
            C = self.config.num_feature_maps + 1 # channel_z + heatmap
            out = conv_block(out, n_filters=C, kernel_size=5, stride=stride_a, use_activation=False)

            z_masked, t = apply_mask(z=out)

            return z_masked, t


    def quantizer(self, z_masked, training):
        def w_times_centers(w):
            matmul_innerproduct = w * self.centers  # (B, C, m, L)
            return tf.reduce_sum(matmul_innerproduct, axis=3)  # (B, C, m)

        HARD_SIGMA = 1e7
        sigma = 1

        with tf.variable_scope('quantizer', reuse=self.reuse_quantizer):
            self.reuse_quantizer = True

            num_centers = self.centers.get_shape().as_list()[-1]

            # reshape (B, C, w, h) to (B, C, m=w*h)
            z_masked_shape = tf.shape(z_masked) # BCwh
            B = z_masked_shape[0]               # B is not necessarily static
            C = int(z_masked.shape[1])          # C is static
            z_masked = tf.reshape(z_masked, [B, C, -1])

            z_masked = tf.expand_dims(z_masked, axis=-1) # make x into (B, C, m, 1)

            dist = tf.square(tf.abs(z_masked - self.centers)) # dist is (B, C, m, L), contains | z_masked_i - c_j | ^ 2

            z_soft = tf.nn.softmax(-sigma * dist, axis=-1)       # (B, C, m, L)
            z_hard = tf.nn.softmax(-HARD_SIGMA * dist, axis=-1)  # (B, C, m, L)

            symbols_hard = tf.argmax(z_hard, axis=-1)
            z_hard = tf.one_hot(symbols_hard, depth=num_centers, axis=-1, dtype=tf.float32)

            z_soft = w_times_centers(z_soft)
            z_hard = w_times_centers(z_hard)

            # reshape to BCwh
            z_soft = tf.reshape(z_soft, z_masked_shape)
            z_hard = tf.reshape(z_hard, z_masked_shape)
            symbols_hard = tf.reshape(symbols_hard, z_masked_shape)

            z_hat = z_soft + tf.stop_gradient(z_hard - z_soft)

            return z_hat, symbols_hard


    def generator(self, z_hat, training):
        def denormalize(x):
            return (x * self.std) + self.mean

        def conv_block(x, n_filters, kernel_size=3, stride=2, use_trasnpose=False, use_activation=True, **kwargs):
            bn_kwargs = {
                'decay': 0.9,
                'epsilon': 1e-5,
                'scale': True,
                'updates_collections': tf.GraphKeys.UPDATE_OPS,
                'fused': True,
                'is_training': training,
                'data_format': 'NCHW'
            }
            conv_kwargs = {
                'weights_regularizer': slim.l2_regularizer(scale=self.config.regularization_factor),
                'data_format': 'NCDHW',
                'normalizer_fn': slim.batch_norm,
                'normalizer_params': bn_kwargs
            }

            if not use_activation:
                conv_kwargs['activation_fn'] = None

            conv = slim.conv3d_transpose if use_trasnpose else slim.conv3d
            
            x = conv(x, n_filters, kernel_size, stride, **conv_kwargs) #[10, 32, 2, 80, 80]

            return x

        @slim.add_arg_scope
        def residual_block(x, n_outputs, n_conv, **kwargs):
            residual_input = x

            for conv_i in range(n_conv):
                if conv_i == (n_conv - 1):  # no relu after final conv
                    kwargs['use_activation'] = False

                x = conv_block(x, n_outputs, **kwargs)

            return x + residual_input

        with tf.variable_scope('generator', reuse=self.reuse_generator):
            self.reuse_generator = True

            n_filters = self.config.n_filters
            d = self.config.depth_size

            out = tf.expand_dims(z_hat, 2)
            out = conv_block(out, n_filters, kernel_size=5, stride=[int(d/4), 2, 2], use_trasnpose=True, use_activation=True)

            res_kwargs = {
                'weights_regularizer': slim.l2_regularizer(scale=self.config.regularization_factor),
                'data_format': 'NCDHW'
            }
            with slim.arg_scope([residual_block], **res_kwargs):
                residual_input_initial = out
                residual_input_b = out

                out = residual_block(out, n_outputs=n_filters, n_conv=2, kernel_size=3, stride=1, use_trasnpose=False, use_activation=True)
                out = residual_block(out, n_outputs=n_filters, n_conv=2, kernel_size=3, stride=1, use_trasnpose=False, use_activation=True)
                out = residual_block(out, n_outputs=n_filters, n_conv=2, kernel_size=3, stride=1, use_trasnpose=False, use_activation=True)

                out = out + residual_input_b

                out = residual_block(out, n_outputs=n_filters, n_conv=2, kernel_size=3, stride=1, use_trasnpose=False, use_activation=False)
                out = out + residual_input_initial
            
            out = conv_block(out, n_filters // 2, kernel_size=5, stride=2, use_trasnpose=True, use_activation=True)
            out = conv_block(out,              1, kernel_size=5, stride=2, use_trasnpose=True, use_activation=False)

            out = denormalize(out)
            out = tf.clip_by_value(out, 0, 1, name='clip')

            return out


    def discriminator(self, x, training, actv=tf.nn.leaky_relu, reuse=False):
        in_kwargs = {'center':True, 'scale':True, 'activation_fn':actv}

        with tf.variable_scope('discriminator', reuse=reuse):
            out = tf.layers.conv3d(x,   64, kernel_size=4, strides=2, padding='SAME', activation=actv)

            out = tf.layers.conv3d(out, 128, kernel_size=4, strides=2, padding='SAME')
            out = actv(tf.contrib.layers.instance_norm(out, **in_kwargs))

            out = tf.layers.conv3d(out, 256, kernel_size=4, strides=2, padding='SAME')
            out = actv(tf.contrib.layers.instance_norm(out, **in_kwargs))

            out = tf.layers.conv3d(out, 512, kernel_size=4, strides=2, padding='SAME')
            out = actv(tf.contrib.layers.instance_norm(out, **in_kwargs))

            out = tf.layers.conv3d(out,  1, kernel_size=4, strides=1, padding='SAME')

            return out


class Probability_Model(object):
    def __init__(self, num_centers):
        self.reuse = False
        self.L = num_centers

        self.first_mask = Utils.create_first_mask()  # DHWio
        self.other_mask = Utils.create_other_mask()  # DHWio

        self.context_size = Utils.get_context_size()


    @property
    def filter_shape(self):
        return 3 // 2 + 1, 3, 3  # CHW


    def bitcost(self, z_hat, target_symbols, training, pad_value):
        """
        Pads q, creates PC network, calculates cross entropy between output of PC network and target_symbols
        :param q: NCHW
        :param target_symbols:
        :param training:
        :return: bitcost per symbol: NCHW
        """
        init = tf.contrib.layers.xavier_initializer()
        init_bias = tf.zeros_initializer()

        def conv_block(name, x, n_outputs, filter_shape, filter_mask, strides=[1,1,1,1,1], use_activation=True):
            n_inputs = x.shape.as_list()[-1]
            filter_shape = tuple(filter_shape) + (n_inputs, n_outputs)

            scope_name = 'conv3d_{}'.format(name)+'_mask'

            with tf.variable_scope(scope_name):
                weights = tf.get_variable('weights', shape=filter_shape, dtype=tf.float32, initializer=init)
                weights = weights * filter_mask

                biases = tf.get_variable('biases', shape=(n_outputs,), dtype=tf.float32, initializer=init_bias)

            x = tf.nn.conv3d(x, weights, strides, padding='VALID', name='conv3d')
            x = tf.nn.bias_add(x, biases, name='bias3d')

            if use_activation:
                x = tf.nn.relu(x)

            return x

        def residual_block(x, n_conv=2, name=None):
            n_outputs = x.shape.as_list()[-1]
            residual_input = x

            with tf.variable_scope(name, 'res'):
                for conv_i in range(n_conv):
                    use_activation = False if conv_i == (n_conv - 1) else True # no relu after final conv
                    x = conv_block('conv{}'.format(conv_i+1), x, n_outputs, self.filter_shape, filter_mask=self.other_mask, use_activation=use_activation)

                return x + residual_input[..., 2:, 2:-2, 2:-2, :]  # for padding

        def padding(x, pad_value):
            """
            :param x: NCHW tensorflow Tensor or numpy array
            """
            pad = self.context_size // 2
            assert pad >= 1

            pads = [
                [0, 0],  # don't pad batch dimension
                [pad, 0],  # don't pad depth_future, it's not seen by any filter
                [pad, pad],
                [pad, pad]
            ]
            assert len(pads) == x.shape.ndims, '{} != {}'.format(len(pads), x.shape)

            return tf.pad(x, pads, constant_values=pad_value)

        def logits(x, training):
            x = conv_block('conv0', x, 24, self.filter_shape, filter_mask=self.first_mask, use_activation=True)
            x = residual_block(x, n_conv=2, name='res1')
            x = conv_block('conv2', x, self.L, self.filter_shape, filter_mask=self.other_mask, use_activation=True)

            return x

        tf_helpers.assert_ndims(z_hat, 4)

        with tf.variable_scope('probability_model', reuse=self.reuse):
            self.reuse = True

            targets_one_hot = tf.one_hot(target_symbols, depth=self.L, axis=-1, name='target_symbols')

            z_pad = padding(z_hat, pad_value=pad_value)

            # make it into NCHWT, where T is the channel dim of the conv3d
            z_pad = tf.expand_dims(z_pad, -1, name='NCHWT')
            logits = logits(z_pad, training)

            if targets_one_hot.shape.is_fully_defined() and logits.shape.is_fully_defined():
                tf_helpers.assert_equal_shape(targets_one_hot, logits)

            # softmax_cross_entropy_with_logits is basis e, change base to 2
            log_base_change_factor = tf.constant(np.log2(np.e), dtype=tf.float32)

            bitcost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets_one_hot) * log_base_change_factor  # NCHW

            return bitcost

