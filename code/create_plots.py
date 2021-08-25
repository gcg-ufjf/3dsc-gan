import matplotlib.pyplot as plt
import numpy as np

def create_plot(output_file, metric, log_dir):
    train=[]
    valid=[]

    with open(output_file, 'r') as file:
        for line in file:
            if metric == 'g_loss':
                if line.split()[0] == 'G_train_loss':
                    train.append(np.float32(line.split()[-1].replace('\n','')))
                if line.split()[0] == 'G_test_loss':
                    valid.append(np.float32(line.split()[-1].replace('\n','')))
            elif metric == 'd_loss':
                if line.split()[0] == 'D_train_loss':
                    train.append(np.float32(line.split()[-1].replace('\n','')))
                if line.split()[0] == 'D_test_loss':
                    valid.append(np.float32(line.split()[-1].replace('\n','')))

            else:
                if line.split()[0] == 'train_'+metric+'':
                    train.append(np.float32(line.split()[-1].replace('\n','')))

                if line.split()[0] == 'test_'+metric+'':
                    valid.append(np.float32(line.split()[-1].replace('\n','')))

    plt.switch_backend('agg')

    fig, ax = plt.subplots()

    smaller_set = len(valid)
    smaller_x = np.arange(smaller_set)

    larger_set = len(train)
    if larger_set > 0:
        larger_x = np.linspace(0, smaller_set-1, retstep = float(smaller_set)/larger_set, num = larger_set)

        plt.plot(larger_x[0], train)
        plt.plot(smaller_x, valid)
        plt.legend(['Train', 'Valid'])
        plt.title('Model ' + metric)
        plt.ylabel(metric)
        plt.xlabel('Epochs')
        fig.savefig(log_dir + '/train_' + metric + '.png')


def create_plots(output_file, log_dir):
    metrics = ['g_loss', 'd_loss', 'psnr', 'mse', 'bpp', 'snr']

    for metric in metrics:
        create_plot(output_file, metric, log_dir)