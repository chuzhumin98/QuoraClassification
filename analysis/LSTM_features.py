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
    path = './LSTM_with_features'
    attri_nums = [0, 10, 20, 50, 100, 200]
    attri_strs = ['n = {}'.format(n) for n in attri_nums]

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
    plt.savefig('image/attributes_loss.png')


    plt.figure(1)
    xs = np.array(range(train_f1s_list.shape[1])) + 1
    for i in range(train_f1s_list.shape[0]):
        plt.plot(xs, train_f1s_list[i,:])
    plt.xlabel('training process')
    plt.ylabel('F1-score')
    plt.title('the F1-score of train data in training process')
    plt.legend(attri_strs)
    plt.savefig('image/attributes_train_F1.png')


    plt.figure(2)
    xs = np.array(range(val_f1s_list.shape[1])) + 1
    for i in range(val_f1s_list.shape[0]):
        plt.plot(xs, val_f1s_list[i, :])
    plt.xlabel('training process')
    plt.ylabel('F1-score')
    plt.title('the F1-score of validate data in training process')
    plt.legend(attri_strs)
    plt.savefig('image/attributes_val_F1.png')

    plt.show()



