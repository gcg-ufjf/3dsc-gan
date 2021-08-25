import os, argparse
import numpy as np
import segyio, shutil

def save_dataset_statistics(output_path, dataset_name, shape, min, max, var, std, mean):

    statistic_path = os.path.join(output_path, 'statistics')
    os.makedirs(statistic_path, exist_ok=True)

    statistic_file = os.path.join(statistic_path, dataset_name + '.metrics')

    with open(statistic_file, 'w+') as s_file:
        s_file.write('Shape: '+str(shape)+'\n')
        s_file.write('Min: '  +str(min)  +'\n')
        s_file.write('Max: '  +str(max)  +'\n')
        s_file.write('Var: '  +str(var)  +'\n')
        s_file.write('Std: '  +str(std)  +'\n')
        s_file.write('Mean: ' +str(mean) +'\n\n')


def generate_dataset(dataset_file, output_path, crop_size, directions):

    print('\nReading file: ' + dataset_file)

    file_without_ext = dataset_file.split('/')[-1].split('.')[0]

    # -------------------------------------------------------------------- #

    new_path = os.path.join(output_path, file_without_ext)
    os.makedirs(new_path, exist_ok=True)
    for dir in directions:
        volume = np.ndarray.astype(np.load(dataset_file), np.float32)

        shape = volume.shape # (n_iline, n_xline, n_depth)

        max = np.max(volume)
        min = np.min(volume)

        if dir == 'inline':
            volume = np.transpose(volume, (2,1,0))
        elif dir == 'crossline':
            volume = np.transpose(volume, (2,0,1))
        elif dir == 'depth':
            volume = np.transpose(volume, (1,0,2))
        else:
            continue

        new_shape = volume.shape
        mod       = tuple(np.mod(new_shape, crop_size))
        padding   = tuple(np.subtract(crop_size, mod))

        volume = np.pad(volume, ((0, padding[0]), (0, padding[1]), (0, padding[2])), 'symmetric')

        cube_count = 0
        for i in range(0, new_shape[0]+padding[0], crop_size[0]):
            for j in range(0, new_shape[1]+padding[1], crop_size[1]):
                for k in range(0, new_shape[2]+padding[2], crop_size[2]):
                    subvolume = volume[i:i+crop_size[0], j:j+crop_size[1], k:k+crop_size[2]]

                    file_name = os.path.join(new_path, file_without_ext+'_'+dir+'_vol_'+str(cube_count)+'_ori.npy')
                    np.save(file_name, subvolume)

                    cube_count += 1

        print('\t', cube_count, dir, 'volumes extracted')

        deep = crop_size[2]
        slice_count = 0
        for i in range(0, new_shape[2]+padding[2], deep):
            subvolume = volume[:, :, i:i+deep]

            file_name = os.path.join(new_path, file_without_ext+'_'+dir+'_vol_slice_'+str(slice_count)+'_ori.npy')
            np.save(file_name, subvolume)

            slice_count += 1

        print('\t', slice_count, dir, 'slices extracted')

        volume = None
        del volume

    # ------------------------------------------------------------------------------------------------ #

    volume = np.load(dataset_file)

    shape = volume.shape

    max = np.max(volume)
    min = np.min(volume)

    mean = std = var = np.float32(0.0)
    inline = np.swapaxes(volume[0], 0, 1)

    for i in range(shape[0]):
        inline = np.swapaxes(volume[i], 0, 1)
        mean += np.sum(inline)

    mean /= (shape[0] * shape[1] * shape[2])

    for i in range(shape[0]):
        inline = np.swapaxes(volume[i], 0, 1)
        var += np.sum((inline - mean) ** 2)

    var /= (shape[0] * shape[1] * shape[2])
    std = np.sqrt(var)

    save_dataset_statistics(output_path, file_without_ext, shape, min, max, var, std, mean)

    # ------------------------------------------------------------------------------------------------ #


def main():
    input_path  = 'data/original_data/'
    output_path = 'data/3D/'
    crop_size   = [160,160,12]
    directions  =['inline', 'crossline']

    crop_folder = ''
    for i in crop_size:
        crop_folder += str(i)+'_'
    crop_folder = crop_folder[:-1]

    new_output_path = os.path.join(output_path, crop_folder)

    shutil.rmtree(new_output_path, ignore_errors=True)

    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith('.npy'):
                dataset_file = os.path.join(root, file)
                generate_dataset(dataset_file, new_output_path, crop_size, directions)


if __name__ == '__main__':
    main()

