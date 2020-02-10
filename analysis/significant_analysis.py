import pickle as pkl
import os

import numpy as np

def save_object(obj,path):
    with open(path,'wb') as files:
        pkl.dump(obj,files)


def load_object(path):
    with open(path,'rb') as files:
        obj = pkl.load(files)
    return obj


def get_significant_prob(val_labels, run1_labels, run2_labels, alpha=(1,1,1), times=100000):
    N_1, N_2, N_3 = 0, 0, 0
    for i in range(len(val_labels)):
        if val_labels[i] == run1_labels[i] and val_labels[i] != run2_labels[i]:
            N_1 += 1
        elif val_labels[i] != run1_labels[i] and val_labels[i] == run2_labels[i]:
            N_2 += 1
        else:
            N_3 += 1

    random_experiments = np.random.dirichlet((N_1+alpha[0], N_2+alpha[1], N_3+alpha[2]), times)

    prob = float(sum(random_experiments[:, 0] > random_experiments[:, 1])) / float(times)
    print(sum(random_experiments[:, 0] > random_experiments[:, 1]))
    return prob



if __name__ == '__main__':
    path = './significant_models'
    file_names = ['bagging.txt', 'BiLSTM.txt', 'mlp.txt']

    val_labels = load_object('../data/valid_label.pkl')
    val_labels = np.array(val_labels)

    # print(val_labels)
    print(len(val_labels))

    results = []

    for i in range(len(file_names)):
        file = open(os.path.join(path, file_names[i]), 'r')
        print(file_names[i])
        while True:
            line = file.readline()
            if line:
                labels = line.split(',')
                labels = [int(label) for label in labels]
                labels = np.array(labels)
                results.append(labels)
                # print(labels)
                print(len(labels))
            else:
                break

        file.close()


    for i in range(len(file_names)):
        print(float(sum(val_labels == results[i])) / float(len(val_labels)))

    def test(index1, index2):
        print('the prob of {} > {} is {}'.format(file_names[index1], file_names[index2],
                                                 get_significant_prob(val_labels, results[index1], results[index2])))

    test(0, 1)
    test(0, 2)
    test(1, 2)


