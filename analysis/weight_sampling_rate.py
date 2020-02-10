import os
import numpy as np
import matplotlib.pyplot as plt

def get_evaluate(path):
    file = open(path, 'r')
    training_losses = file.readline().split(',')[1:]
    training_losses = [float(training_loss) for training_loss in training_losses]

    training_f1s = file.readline().split(',')[1:]
    training_f1s = [float(training_f1) for training_f1 in training_f1s]

    val_f1s = file.readline().split(',')[1:]
    val_f1s = [float(val_f1) for val_f1 in val_f1s]

    print(len(training_losses), len(training_f1s), len(val_f1s))
    # print(training_losses)
    # print(training_f1s)
    # print(val_f1s)
    file.close()

    return training_losses, training_f1s, val_f1s


if __name__ == '__main__':
    path = './weight_sampling_rate'
    attri_nums = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
    attri_strs = ['rate = {}'.format(n) for n in attri_nums]

    best_f1s = [0.675, 0.683, 0.675, 0.655, 0.647, 0.649]

    train_losses_list = []
    train_f1s_list = []
    val_f1s_list = []

    for i in range(len(attri_nums)):
        file_name = os.path.join(path, '{}.txt').format(attri_nums[i])
        train_losses, train_f1s, val_f1s = get_evaluate(file_name)
        train_losses_list.append(train_losses)
        train_f1s_list.append(train_f1s)
        val_f1s_list.append(val_f1s)

    train_losses_list = np.array(train_losses_list)
    train_f1s_list = np.array(train_f1s_list)
    val_f1s_list = np.array(val_f1s_list)

    if not os.path.exists('image'):
        os.makedirs('image')

    plt.figure(0)
    xs = np.array(range(train_losses_list.shape[1])) + 1
    for i in range(train_losses_list.shape[0]):
        plt.plot(xs, train_losses_list[i,:])
    plt.legend(attri_strs)
    plt.xlabel('training process')
    plt.ylabel('loss')
    plt.title('the train loss in training process')
    plt.savefig('image/rate_loss.png')


    plt.figure(1)
    xs = np.array(range(train_f1s_list.shape[1])) + 1
    for i in range(train_f1s_list.shape[0]):
        plt.plot(xs, train_f1s_list[i,:])
    plt.xlabel('training process')
    plt.ylabel('F1-score')
    plt.title('the F1-score of train data in training process')
    plt.legend(attri_strs)
    plt.savefig('image/rate_train_F1.png')


    plt.figure(2)
    xs = np.array(range(val_f1s_list.shape[1])) + 1
    for i in range(val_f1s_list.shape[0]):
        plt.plot(xs, val_f1s_list[i, :])
    plt.xlabel('training process')
    plt.ylabel('F1-score')
    plt.title('the F1-score of validate data in training process')
    plt.legend(attri_strs)
    plt.savefig('image/rate_val_F1.png')


    plt.figure(3)
    xs = np.array(range(len(attri_nums)))
    plt.plot(xs, best_f1s, 'b', marker='.', markersize=10, lw=1.5)
    for i in range(len(xs)):
        if i == 0:
            plt.text(xs[i] + 0.04, best_f1s[i] - 0.0014, str(best_f1s[i]))
        elif i == 4:
            plt.text(xs[i] - 0.04, best_f1s[i] + 0.001, str(best_f1s[i]))
        elif i == 5:
            plt.text(xs[i] - 0.2, best_f1s[i] + 0.0006, str(best_f1s[i]))
        else:
            plt.text(xs[i]+0.04, best_f1s[i]+0.0004, str(best_f1s[i]))
    plt.xticks(xs, attri_strs)
    plt.title('best validate F1-score vs rate')
    # plt.grid()

    plt.savefig('image/rate_best_f1.png')


    plt.show()
