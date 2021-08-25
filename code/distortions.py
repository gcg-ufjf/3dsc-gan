import tensorflow as tf
from fjcommon import tf_helpers


class Distortions(object):
    def __init__(self, x, x_out):
        assert tf.float32.is_compatible_with(x.dtype) and tf.float32.is_compatible_with(x_out.dtype)

        squared_error = tf.square(x_out - x)
        mse_per_image = tf.reduce_mean(squared_error, axis=[1, 2, 3, 4])
        psnr_per_image = -10 * tf_helpers.log10(mse_per_image)

        self.mse = tf.reduce_mean(mse_per_image)
        self.psnr = tf.reduce_mean(psnr_per_image)

        self.d_loss = self.get_distortion()

    def get_distortion(self):
        return 100 - self.psnr