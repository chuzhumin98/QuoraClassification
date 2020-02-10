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
    path = './train_process'
    attri_nums = ['GRU', 'LSTM', 'BiLSTM', 'BiLSTM+Attention']
    attri_strs = attri_nums

    train_losses_list = []
    train_f1s_list = []
    val_f1s_list = []
    split_num = 10
    cut_off = 350

    for i in range(len(attri_nums)):
        file_name = os.path.join(path, '{}.txt').format(attri_nums[i])
        train_losses, train_f1s, val_f1s = get_evaluate(file_name)
        train_losses = [train_losses[(i+1)*split_num-1] for i in range(len(train_losses)//split_num)]
        train_losses = np.array(train_losses[:cut_off])
        train_losses_list.append(train_losses)
        train_f1s_list.append(train_f1s)
        val_f1s_list.append(val_f1s)

    train_losses_list = np.array(train_losses_list)
    train_f1s_list = np.array(train_f1s_list)
    val_f1s_list = np.array(val_f1s_list)

    print(train_losses_list.shape)

    if not os.path.exists('image'):
        os.makedirs('image')

    fig = plt.figure(1)
    xs = np.array(range(train_f1s_list.shape[1])) + 1
    ax1 = fig.add_subplot(111)
    for i in range(train_f1s_list.shape[0]):
        ax1.plot(xs, train_f1s_list[i,:])
    ax1.set_ylabel('F1-score')
    ax1.set_title("the train F1-score and loss in training process")

    ax2 = ax1.twinx()  # this is the important function
    for i in range(train_losses_list.shape[0]):
        ax2.plot(xs, train_losses_list[i,:])
    ax2.set_ylabel('loss')
    ax2.set_xlabel('training process')
    plt.legend(attri_strs, loc=7)
    plt.savefig('image/models_train_results.png')



    plt.figure(2)
    xs = np.array(range(val_f1s_list.shape[1])) + 1
    for i in range(val_f1s_list.shape[0]):
        plt.plot(xs, val_f1s_list[i, :])
    plt.xlabel('training process')
    plt.ylabel('F1-score')
    plt.title('the F1-score of validate data in training process')
    plt.legend(attri_strs)
    plt.savefig('image/models_val_F1.png')

    plt.show()



