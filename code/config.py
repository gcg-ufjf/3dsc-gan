class config_model(object):
    batch_size = 8
    epochs = 60

    depth_size = 12
    n_filters = 128

    centers_initial_range = (-10,10)
    num_centers = 50

    regularization_factor_centers = 0.01
    regularization_factor = 0.005

    num_feature_maps = 64

    target_bpv = 0.25
    beta = 100
    H_target = target_bpv * depth_size


class config_test(object):
    direction = 'inline' # crossline or depth
    dataset_name = 'Penobscot3D'


class directories(object):
    log_dir = 'logs/'
    checkpoints = 'logs/checkpoints/'

    train_filename = 'datasets/train_scale_12'
    test_filename = 'datasets/test_scale_12'

    datasets_path = '../data/3D/160_160_12/'

    statistics_file = 'statistics/pari+f3'
    statistics_path = '../data/3D/160_160_12/statistics/'