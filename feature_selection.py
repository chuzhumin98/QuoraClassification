import pickle as pkl

from dataloader import Worddict
from dataloader import save_object

import numpy as np
import os

import math

def get_appear_info(sentences, labels, word2idx, idx2word, is_train=True, pathes=None):
    if pathes is None:
        size = len(idx2word)
        freq_per_label = np.zeros([2, size])
        appeardoc_per_label = np.zeros([2, size])

        sentence_len = [[], []]

        for i in range(len(sentences)):
            sentence = sentences[i]
            index = labels[i]
            words = sentence.strip().split()
            word_set = set()
            for word in words:
                if word in word2idx:
                    freq_per_label[index, word2idx[word]] += 1
                    if word not in word_set:
                        word_set.add(word)
                        appeardoc_per_label[index, word2idx[word]] += 1
                else:
                    freq_per_label[index, word2idx['<unk>']] += 1
                    if '<unk>' not in word_set:
                        word_set.add('<unk>')
                        appeardoc_per_label[index, word2idx['<unk>']] += 1
            sentence_len[index].append(len(words))

        if not os.path.exists('./cache'):
            os.makedirs('./cache')

        if is_train:
            type = 'train'
        else:
            type = 'valid'
        save_object(appeardoc_per_label, './cache/{}_appeardoc_per_label.pkl'.format(type))
        save_object(freq_per_label, './cache/{}_freq_per_label.pkl'.format(type))
        save_object(sentence_len, './cache/{}_sentence_len.pkl'.format(type))

    else:

        with open(pathes[0], 'rb') as file:
            appeardoc_per_label = pkl.load(file)

        with open(pathes[1], 'rb') as file:
            freq_per_label = pkl.load(file)

        with open(pathes[2], 'rb') as file:
            sentence_len = pkl.load(file)

    print(freq_per_label)
    print(np.sum(freq_per_label, axis=1))
    print(np.sum(freq_per_label, axis=1) / np.array([len(sentences) - np.sum(labels), np.sum(labels)]))
    print(np.std(np.array(sentence_len[0])), np.std(np.array(sentence_len[1])))
    print(appeardoc_per_label)
    print(np.sum(appeardoc_per_label, axis=1))

    print(idx2word)

    return appeardoc_per_label, freq_per_label, sentence_len

def entropy(N, n):
    # group N to n and (N - n)
    if n == 0 or N == n:
        return 0
    else:
        p = float(n) / float(N)
        # print('p = {}'.format(p))
        return -p*math.log2(p)-(1-p)*math.log2(1-p)


def calculate_IG(appeardoc_per_label, total_docs):
    # calculate for information gain
    size = appeardoc_per_label.shape[1]
    entropy0 = entropy(sum(total_docs), total_docs[0])
    entropys = np.zeros([size])
    for i in range(size):
        p_0 = float(sum(appeardoc_per_label[:,i])) / float(sum(total_docs))
        entropys[i] =  p_0*(entropy0 - entropy(sum(appeardoc_per_label[:, i]), appeardoc_per_label[0, i])) + \
                      (1-p_0)*(entropy0 - entropy(sum(total_docs)-sum(appeardoc_per_label[:,i]), total_docs[0]-appeardoc_per_label[0,i]))

    print(entropys[:1000])
    return entropys

def array2dict(words):
    words_dict = {}
    for i in range(len(words)):
        words_dict[words[i]] = i

    return words_dict

def get_selected_words(appeardoc_per_label, total_docs, cutoff=100, path=None):
    if path is None:
        entropys = calculate_IG(appeardoc_per_label, total_docs)

        order = np.argsort(-entropys)


        print('following is the top IG words:')

        selected_words = np.array(idx2word)[order[:cutoff]]

        print('max entropy:',entropys[order[0]])

        # print(entropys[order[:cutoff]])

        if not os.path.exists('./cache'):
            os.makedirs('./cache')

        save_object(selected_words, './cache/selected_words_{}.pkl'.format(cutoff))
    else:
        with open('./cache/selected_words_{}.pkl'.format(cutoff), 'rb') as file:
            selected_words = pkl.load(file)

    print(selected_words)

    return selected_words


def get_attribute(data_path, is_train, is_test, cutoff=100):
    with open(data_path, 'rb') as file:
        sentences = pkl.load(file)

    attributes = np.zeros([len(sentences), cutoff])

    for i in range(len(sentences)):
        sentence = sentences[i]
        words = sentence.strip().split()
        for word in words:
            if word in selected_words_dict:
                attributes[i, selected_words_dict[word]] = 1.

    print(attributes)

    if not os.path.exists('./data'):
        os.makedirs('./data')

    if is_train:
        save_object(attributes, './data/train_words_attributes_{}.pkl'.format(cutoff))
    elif not is_test:
        save_object(attributes, './data/valid_words_attributes_{}.pkl'.format(cutoff))
    else:
        save_object(attributes, './data/test_words_attributes_{}.pkl'.format(cutoff))

    return attributes



if __name__ == '__main__':
    with open('data/train_data.pkl', 'rb') as file:
        sentences = pkl.load(file)
    with open('data/train_label.pkl', 'rb') as file2:
        labels = pkl.load(file2)

    '''
    for i in range(1000):
        if sentences[i].find('math') >= 0 or sentences[i].find('<n>') >= 0 or sentences[i].find('<time>') >=0:
            print(sentences[i])
    '''

    with open('data/word_dict.pkl', 'rb') as file3:
        word_dict = pkl.load(file3)

    word2idx = word_dict.word2idx
    idx2word = word_dict.idx2word

    pathes = ['./cache/train_appeardoc_per_label.pkl', './cache/train_freq_per_label.pkl', './cache/train_sentence_len.pkl']
    # first run change to pathes = None
    appeardoc_per_label, freq_per_label, sentence_len = get_appear_info(sentences, labels, word2idx, idx2word, is_train=True, pathes=pathes)


    total_docs = [len(sentences) - np.sum(labels), np.sum(labels)]
    print('total_docs:{}'.format(total_docs))

    cutoff = 100

    # first run change to path = None
    selected_words = get_selected_words(appeardoc_per_label, total_docs, cutoff=cutoff, path=None) #'./cache/selected_words_100.pkl')
    selected_words_dict = array2dict(selected_words)

    # get_attribute('data/train_data.pkl', is_train=True, is_test=False, cutoff=cutoff)
    # get_attribute('data/valid_data.pkl', is_train=False, is_test=False, cutoff=cutoff)
    # get_attribute('data/test_cleaned_questions.pkl', is_train=False, is_test=True, cutoff=cutoff)


















