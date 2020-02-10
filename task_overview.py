from dataloader import load_object
from dataloader import Worddict

import matplotlib.pyplot as plt
import seaborn

import numpy as np
import pandas as pd
import os




if __name__ == '__main__':
    '''
    path = 'data/train.csv'
    file = open(path, 'r')
    line = file.readline()
    cnt = 0
    cnt_count1 = 0
    idx = 0
    while True:
        line = file.readline()
        if line:
            try:
                qid, content = line.split(',', 1)
                label = int(content.split(',')[-1])
                cnt += 1
                cnt_count1 += label
            except:
                print(line.strip())
                idx = 1
            if idx == 2:
                print(line.strip())
                print()
                idx = 0
            if idx == 1:
                idx = 2

        else:
            break

    print('train dataset number: {}'.format(cnt))
    print('in this label 0 number: {}'.format(cnt - cnt_count1))
    print('in this label 1 number: {}'.format(cnt_count1))

    file.close()
    '''

    path_question = 'data/cleaned_questions.pkl'
    path_labels = 'data/labels.pkl'

    questions = load_object(path_question)
    labels = load_object(path_labels)
    print(len(questions))
    print(len(labels))

    lengths = [len(question.split()) for question in questions]
    print(lengths)

    # len_sort = sorted(lengths)
    # print(len_sort[0], len_sort[-1])

    lengths = np.array(lengths)

    if not os.path.exists('image'):
        os.makedirs('image')

    # print(sum(lengths > 60))
    # lengths = lengths[lengths < 60]
    # plt.figure(1)
    # plt.hist(lengths, bins=20, color='b')
    # plt.xlabel('length of question')
    # plt.ylabel('samples number')
    # plt.title("hist of the questions' length")
    # plt.savefig('image/hist_of_questions.png')

    # labels = np.array(labels)
    # lengths_0 = lengths[labels == 0]
    # lengths_1 = lengths[labels == 1]
    # print(len(lengths_0), len(lengths_1))
    #
    # s_0 = pd.Series(lengths_0)
    # s_1 = pd.Series(lengths_1)
    # data = pd.DataFrame({'label=0': s_0, 'label=1': s_1})
    # plt.figure(2)
    # seaborn.boxplot(data=data)
    # plt.ylim(0,60)
    # plt.ylabel('length of questions')
    # plt.title("boxplot of the questions' length in each label")
    # plt.savefig('./image/boxplot_questions.png')


    # for i in range(len(questions)):
    #     # if questions[i].find('<n>') >= 0:
    #     #     print(questions[i])
    #
    #     if questions[i].find('<url>') >= 0:
    #         print(questions[i])
    #
    #     if questions[i].find('math') >= 0:
    #         print(questions[i])
    #         print(i)
    #
    #     if questions[i].find('<time>') >= 0:
    #         print(questions[i])
    #         print(i)

    word_dict = load_object('data/word_dict_(randn).pkl')
    print(word_dict.word_num)

