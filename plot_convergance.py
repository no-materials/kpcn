# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#
# Common libs
import numpy as np
import matplotlib.pyplot as plt
from os.path import isfile, join, exists
from os import listdir

# My libs
from utils.config import Config


def running_mean(signal, n, axis=0):
    signal = np.array(signal)
    if signal.ndim == 1:
        signal_sum = np.convolve(signal, np.ones((2 * n + 1,)), mode='same')
        signal_num = np.convolve(signal * 0 + 1, np.ones((2 * n + 1,)), mode='same')
        return signal_sum / signal_num

    elif signal.ndim == 2:
        smoothed = np.empty(signal.shape)
        if axis == 0:
            for i, sig in enumerate(signal):
                sig_sum = np.convolve(sig, np.ones((2 * n + 1,)), mode='same')
                sig_num = np.convolve(sig * 0 + 1, np.ones((2 * n + 1,)), mode='same')
                smoothed[i, :] = sig_sum / sig_num
        elif axis == 1:
            for i, sig in enumerate(signal.T):
                sig_sum = np.convolve(sig, np.ones((2 * n + 1,)), mode='same')
                sig_num = np.convolve(sig * 0 + 1, np.ones((2 * n + 1,)), mode='same')
                smoothed[:, i] = sig_sum / sig_num
        else:
            print('wrong axis')
        return smoothed

    else:
        print('wrong dimensions')
        return None


def load_training_results(path):
    filename = join(path, 'training.txt')
    epochs = []
    steps = []
    L_out = []
    L_reg = []
    L_p = []
    coarse_EM = []
    fine_CD = []
    mixed_loss = []
    t = []
    memory = []
    with open(filename, 'r') as f:
        for line in f:
            line_info = line.split()
            if len(line) > 0:
                try:
                    epochs += [int(line_info[0])]
                    steps += [int(line_info[1])]
                    L_out += [float(line_info[2])]
                    L_reg += [float(line_info[3])]
                    L_p += [float(line_info[4])]
                    coarse_EM += [float(line_info[5])]
                    fine_CD += [float(line_info[6])]
                    mixed_loss += [float(line_info[7])]
                    t += [float(line_info[8])]
                    memory += [float(line_info[9])]
                except ValueError as e:
                    print("error", e, "on line", epochs[-1])
            else:
                break

    return steps, L_out, L_reg, L_p, coarse_EM, fine_CD, mixed_loss, t, memory


def load_validation_results(path):
    filename = join(path, 'validation.txt')
    with open(filename, 'r') as f:
        lines = f.readlines()

    epochs = []
    steps = []
    coarse_EM = []
    fine_CD = []
    mixed_loss = []
    for line in lines[1:]:
        line_info = line.split()
        if len(line) > 0:
            epochs += [int(line_info[0])]
            steps += [int(line_info[1])]
            coarse_EM += [float(line_info[2])]
            fine_CD += [float(line_info[3])]
            mixed_loss += [float(line_info[4])]
        else:
            break

    return epochs, steps, coarse_EM, fine_CD, mixed_loss


def compare_trainings(list_of_paths, list_of_labels=None):
    # Parameters
    # **********

    steps_per_epoch = 0
    smooth_epochs = 1

    if list_of_labels is None:
        list_of_labels = [str(i) for i in range(len(list_of_paths))]

    # Read Training Logs
    # ******************

    all_epochs = []
    all_loss = []
    all_lr = []
    all_times = []

    for path in list_of_paths:

        # Load parameters
        config = Config()
        config.load(path)

        # Compute number of steps per epoch
        if config.epoch_steps is None:
            if config.dataset == 'ModelNet40':
                steps_per_epoch = np.ceil(9843 / int(config.batch_num))
            else:
                raise ValueError('Unsupported dataset')
        else:
            steps_per_epoch = config.epoch_steps

        if config.dataset == 'ShapeNetV1':
            steps_per_epoch = np.ceil(57946 / int(config.batch_num))  # 3622
        elif config.dataset == 'pc_shapenetCompletionBenchmark2048':
            steps_per_epoch = np.ceil(28974 / int(config.batch_num))  # 1810,...

        smooth_n = int(steps_per_epoch * smooth_epochs)

        # Load results
        steps, L_out, L_reg, L_p, coarse_EM, fine_CD, mixed_loss, t, memory = load_training_results(path)
        # epochs, steps, coarse_EM, fine_CD, mixed_loss = load_validation_results(path)
        all_epochs += [np.array(steps) / steps_per_epoch]
        all_loss += [running_mean(mixed_loss, smooth_n)]
        all_times += [t]

        all_epochs2 = all_epochs[0][0:360392]
        all_loss2 = all_loss[0][0:360392]

        # Learning rate
        lr_decay_v = np.array([lr_d for ep, lr_d in config.lr_decays.items()])
        lr_decay_e = np.array([ep for ep, lr_d in config.lr_decays.items()])
        max_e = max(np.max(all_epochs[-1]) + 1, np.max(lr_decay_e) + 1)
        lr_decays = np.ones(int(np.ceil(max_e)), dtype=np.float32)
        lr_decays[0] = float(config.learning_rate)
        lr_decays[lr_decay_e] = lr_decay_v
        lr = np.cumprod(lr_decays)
        all_lr += [lr[np.floor(all_epochs[-1]).astype(np.int32)]]

    # Plots learning rate
    # *******************

    if False:
        # Figure
        fig = plt.figure('lr')
        for i, label in enumerate(list_of_labels):
            plt.plot(all_epochs[i], all_lr[i], linewidth=1, label=label)

        # Set names for axes
        plt.xlabel('epochs')
        plt.ylabel('lr')
        plt.yscale('log')

        # Display legends and title
        plt.legend(loc=1)

        # Customize the graph
        ax = fig.gca()
        ax.grid(linestyle='-.', which='both')
        # ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Plots loss
    # **********

    # Figure
    fig = plt.figure('loss')
    for i, label in enumerate(list_of_labels):
        plt.plot(all_epochs2, all_loss2, linewidth=1, label=label)

    # Set names for axes
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.yscale('log')

    # Display legends and title
    plt.legend(loc=1)
    plt.title('Training Loss')

    # Customize the graph
    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')
    # ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Plot Times
    # **********

    # Figure
    # fig = plt.figure('time')
    # for i, label in enumerate(list_of_labels):
    #     plt.plot(all_epochs[i], np.array(all_times[i]) / 3600, linewidth=1, label=label)
    #
    # # Set names for axes
    # plt.xlabel('epochs')
    # plt.ylabel('time')
    # # plt.yscale('log')
    #
    # # Display legends and title
    # plt.legend(loc=0)
    #
    # # Customize the graph
    # ax = fig.gca()
    # ax.grid(linestyle='-.', which='both')
    # # ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Show all
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#


if __name__ == '__main__':
    ######################################################
    # Choose a list of log to plot together for comparison
    ######################################################

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-12-08_14-44-40'
    end = 'Log_2019-12-08_14-44-40'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])

    # Give names to the logs (for legends)
    logs_names = ['Log_SN2048_batch16']
    logs_names = np.array(logs_names[:len(logs)])

    ################################################################
    # The right plotting function is called depending on the dataset
    ################################################################

    # Plot the training loss and accuracy
    compare_trainings(logs, logs_names)
