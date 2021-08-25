import os
import numpy as np
import tensorflow as tf

class Utils(object):
    @staticmethod
    def scope_variables(name):
        with tf.variable_scope(name):
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

    @staticmethod
    def get_shape_and_crop_size(shape, direction, file_path):
        assert direction in ['inline', 'crossline', 'depth'], 'Invalid direction'

        n_iline, n_xline, n_depth = shape
        
        if direction == 'inline':
            line_a = n_iline
            line_b = n_depth
            line_c = n_xline

        elif direction == 'crossline':
            line_a = n_xline
            line_b = n_depth
            line_c = n_iline

        else: # direction == 'depth':
            line_a = n_depth
            line_b = n_xline
            line_c = n_iline

        file = np.load(file_path)
        crop_size = np.transpose(file, (2, 0, 1)).shape

        return line_a, line_b, line_c, crop_size

    @staticmethod
    def get_input_shape(batch_size, filename):
        shape = np.load(filename).shape
        return (batch_size, 1, shape[2], shape[0], shape[1])

    @staticmethod
    def get_checkpoint_path(log_dir_root):
        for i in os.listdir(log_dir_root):
            if os.path.isdir(os.path.join(log_dir_root, i)) and ' cvpr@' in i:
                checkpoint_path = os.path.join(log_dir_root, i, 'ckpts')
                log_path = os.path.dirname(checkpoint_path)

                return checkpoint_path, log_path

    @staticmethod
    def get_statistics(statistics, dataset_name=None):
        mean = std = shape = None

        if dataset_name:
            statistics = os.path.join(statistics, dataset_name) + '.metrics'

        with open(statistics) as metrics_file:
            lines = metrics_file.readlines()

            before_norm = False
            after_norm = False
            for line in lines:
                if 'Shape' in line:
                    shape = eval(line.split(': ')[-1])

                if 'Mean' in line:
                    aux = np.float32(line.split(': ')[-1])
                    mean = np.array([aux], dtype=np.float32)

                if 'Std' in line:
                    aux = np.float32(line.split(': ')[-1])
                    std = np.array([aux], dtype=np.float32)

        return mean, std, shape

    @staticmethod
    def create_centers_variable(config):  # (C, L) or (L,)

        def get_centers_initializer(config):
            minval, maxval = map(int, config.centers_initial_range)
            return tf.random_uniform_initializer(minval=minval, maxval=maxval)

        assert config.num_centers is not None
        centers = tf.get_variable(
            'centers', shape=(config.num_centers,), 
            dtype=tf.float32,
            initializer=get_centers_initializer(config)
        )

        return centers

    @staticmethod
    def create_centers_regularization_term(config, centers):
        reg = tf.to_float(config.regularization_factor_centers)
        centers_reg = tf.identity(reg * tf.nn.l2_loss(centers), name='l2_regularizer')
        tf.losses.add_loss(centers_reg, tf.GraphKeys.REGULARIZATION_LOSSES)

    @staticmethod
    def create_first_mask():
        def filter_shape():
            return 3 // 2 + 1, 3, 3  # CHW

        filter_shape = filter_shape()

        # mask is DHW
        mask = np.ones(filter_shape, dtype=np.float32)

        # zero out D = 1
        mask[-1, 3 // 2, 3 // 2:] = 0 # everything to the right of the central pixel, including the central pixel
        mask[-1, 3 // 2 + 1:, :] = 0 # all rows below the central row

        mask = np.expand_dims(np.expand_dims(mask, -1), -1)  # Make into DHWio, for broadcasting with 3D filters

        # make tf conv3d mask
        assert mask.ndim == 5, 'Expected DHWio'
        mask = tf.constant(mask)
        mask = tf.stop_gradient(mask)

        return mask

    @staticmethod
    def create_other_mask():
        def filter_shape():
            return 3 // 2 + 1, 3, 3  # CHW

        filter_shape = filter_shape()

        # mask is DHW
        mask = np.ones(filter_shape, dtype=np.float32)

        # zero out D = 1
        mask[-1, 3 // 2, 3 // 2 + 1:] = 0 # everything to the right of the central pixel, except the central pixel
        mask[-1, 3 // 2 + 1:, :] = 0 # all rows below the central row

        mask = np.expand_dims(np.expand_dims(mask, -1), -1)  # Make into DHWio, for broadcasting with 3D filters

        # make tf conv3d mask
        assert mask.ndim == 5, 'Expected DHWio'
        mask = tf.constant(mask)
        mask = tf.stop_gradient(mask)

        return mask

    @staticmethod
    def get_context_size():
        num_conv = 2
        per_residual = 2
        kernel_size = 3

        num_layers = num_conv + 1 * per_residual
        context_size = num_layers * (kernel_size - 1) + 1

        return context_size

    @staticmethod
    def bitcost_to_bpv(x, bitcost_value):
        assert bitcost_value.shape.ndims == 4
        assert x.shape.ndims == 5
        assert int(x.shape[1]) == 1, 'Expected N1DHW, got {}'.format(x)

        num_bits = tf.reduce_sum(bitcost_value, name='num_bits')
        num_voxel_x = tf.to_float(tf.reduce_prod(tf.shape(x)))

        return num_bits / num_voxel_x

    @staticmethod
    def save_img(img_name, img_out, out_dir):
        assert img_name.endswith('.npy')
        assert img_out.ndim == 5 , 'Expected NCHW, got {}'.format(img_out.shape)

        img_dir = os.path.join(out_dir, 'imgs')
        os.makedirs(img_dir, exist_ok=True)

        img_out = np.transpose(img_out[0, 0, :, :, :], (1, 2, 0))  # Make NCDHW - > HWD
        img_out_p = os.path.join(img_dir, img_name)

        np.save(img_out_p, img_out)

    @staticmethod
    def init_output_file(args, output_file, epochs, epoch_size):
        with open(output_file, 'a') as out:
            with open(args.train_filename) as file:
                out.write('dataset_train:\n')
                for line in file:
                    out.write('\t'+line+'\n')

            with open(args.test_filename) as file:
                out.write('dataset_test:\n')
                for line in file:
                    out.write('\t'+line+'\n')

            out.write('epochs: '+str(epochs)+'\n')
            out.write('total_steps: '+str(epochs * epoch_size)+'\n')

    @staticmethod
    def write_results(log_dir, step, otp, training):
        mode = 'train' if training else 'test'

        with open(log_dir + '/output.txt', 'a') as file:
            file.write('itr = '+str(step)+'\n')
            file.write('G_'+mode+'_loss = '+str(otp['G_loss'])+'\n')
            file.write('D_'+mode+'_loss = '+str(otp['D_loss'])+'\n')
            file.write(mode+'_mse = '+str(otp['mse'])+'\n')
            file.write(mode+'_psnr = '+str(otp['psnr'])+'\n')
            file.write(mode+'_bpv = '+str(otp['bpv'])+'\n')
            file.write('H_'+mode+' = '+str(otp['H_real']) + '\n')
            if not training:
                file.write('centers = '+str(otp['centers']) + '\n')

    @staticmethod
    def print_results(step, otp, training):
        mode = 'train' if training else 'test'

        print('G_'+mode+'_loss = '+str(otp['G_loss']))
        print('D_'+mode+'_loss = '+str(otp['D_loss']))
        print(mode+'_mse = '+str(otp['mse']))
        print(mode+'_psnr = '+str(otp['psnr']))
        print(mode+'_bpv = '+str(otp['bpv']))
        print('H_'+mode+' = '+str(otp['H_real']))
