import os
import numpy as np
import time
import re
import pickle as pkl
from torch.utils.data import Dataset
import torch
import torch.nn as nn

import mosestokenizer

puncnormalize = mosestokenizer.MosesPunctuationNormalizer("en")
normalize = mosestokenizer.MosesTokenizer("en")

import os

def read_in_file(path, labeled=True):
    file = open(path, 'r', encoding='utf-8')
    qids = []
    questions = []
    labels = []
    line = file.readline()
    print(line)
    cnt = 0
    while True:
        line = file.readline()
        if line:
            try:
                qid, question = line.split(',', 1)
                if labeled:
                    question_list = question.strip().split(',')
                    label = int(question_list[-1])
                    question = ','.join(question_list[:-1])
                    
                    labels.append(label)

                qids.append(qid.strip())
                questions.append(question)

                if cnt < 50:
                    if labeled:
                        print('qid:{}\nquestion:{}\nlabel:{}\n'.format(qid, question, labels[-1]))
                    else:
                        print('qid:{}\nquestion:{}\n'.format(qid, question))

                    cnt += 1
            except:
                print(line)
        else:
            break
    print(len(labels), len(qids), len(questions))
    print(sum(labels))
    # for i in range(len(labels)):
    #     if labels[i] == 1:
    #         print(questions[i])
    file.close()
    return qids,labels,questions

def norm(sentence):
    if type(sentence) == str:
        sentence = puncnormalize(sentence)
        sentence = normalize(sentence)
        return sentence
    elif type(sentence) == list:
        return [norm(s) for s in sentence]


def clean_sentences(questions):
    cleaned=[]
    puncs = ["\"", "?", ".",",","(",")","/","\'","!"]
    time_set = set(['<n>s', '<n>-<n>-<n>', '<n>-<n>', '<n>:<n>', '<n>-month', '<n>-year', '<n>-hour', '<n>sec',
                    '<n>-day', '<n>year', '<n>years', '<n>min', '<n>-<n>years', '<n>hour', '<n>day', 'year<n>-<n>',
                    '<n>-months'])
    count=0
    re_nums = []
    re_urls = []
    re_nums.append(re.compile(u'[0-9．·]+'))
    re_nums.append(re.compile(u'[0-9．·]+K'))
    re_nums.append(re.compile(u'[0-9．·]+M'))
    re_nums.append(re.compile(u'[0-9．·]+G'))
    re_nums.append(re.compile(u'[0-9．·]+s'))
    re_urls.append(re.compile(u'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'))
    # re_punc=re.compile(r'[\"?.]')

    for question in questions:
        question=question.lstrip("\"").rstrip("\"")

        try:
            question = " ".join(norm(question))
        except:
            print(question)

        for re_num in re_nums:
            question=re.sub(re_num,' <N> ',question)
        for re_url in re_urls:
            if re.match(re_url,question) is not None:
                question = re.sub(re_url, " <url> ", question)
                print(question)
            # question=re.sub(re_url,"<url>",question)
        for punc in puncs:
            question=question.replace(punc," "+punc+" ")


        question=question.lower()

        words = question.strip().split()
        for i in range(len(words)):
            word = words[i]
            if word in time_set:
                word = '<time>'
            elif word.find('<n>') >= 0 and word != '<n>':
                word = '<math>'

            words[i] = word

        question = ' '.join(words)

        cleaned.append(question)
        count+=1
        if count % 10000 == 0:
            print(count)
            # if count==100000:
            #     break
    return cleaned


class Worddict(object):
    def __init__(self):
        self.word2idx={}
        self.idx2word=[]
        self.idx2freq=[]
        self.word_num=0

    def add_word(self,word,freq=1):
        if word in self.word2idx:
            self.idx2freq[self.word2idx[word]] += 1
        else:
            self.word2idx[word] = self.word_num
            self.idx2freq.append(freq)
            self.idx2word.append(word)
            self.word_num += 1


    def build_embedding_from_pretrained(self, pretrained_source, embedding_dim=300):
        self.has_word_embedding = dict() # to note if a word has an embedding
        for word in self.idx2word:
            self.has_word_embedding[word] = 0

        with open(pretrained_source, 'r') as f:
            start_time = time.time()
            #first filter words
            for ind, line in enumerate(f):
                try:
                    parseline = line.strip().split()
                    #embedding = [float(item) for item in parseline[-embedding_dim:]]
                    word = " ".join(parseline[:-embedding_dim])
                    if word in self.word2idx.keys():
                        #embedding_tensor[self.word2idx[word]] = torch.Tensor(embedding)
                        self.has_word_embedding[word] = 1
                    
                except:
                    print(line)

                finally:
                    if (ind + 1) % 10000 == 0:
                        dur_time = time.time() - start_time
                        num_in_dict = sum(self.has_word_embedding.values())
                        print("%s in source, %s / %s in dict, time: %.2f s" % (ind + 1, num_in_dict, self.word_num, dur_time))
                        start_time = time.time()

        newword2idx = dict()
        newidx2word = []
        newidx2freq = []
        newword_num = 0
        unk_freq = 0
        pad_freq = self.idx2freq[self.word2idx['<pad>']] 
        #url_freq = self.idx2freq[self.word2idx['<url>']] 
        for word, retain in self.has_word_embedding.items():
            if retain:
                newword2idx[word] = newword_num
                newidx2word.append(word)
                newidx2freq.append(self.idx2freq[self.word2idx[word]])
                newword_num += 1
            else:
                if not word in ['<n>', '<pad>', '<url>']:
                    unk_freq += self.idx2freq[self.word2idx[word]]

        self.word2idx = newword2idx
        self.idx2word = newidx2word
        self.idx2freq = newidx2freq
        self.word_num = len(self.idx2word)
        self.add_word("<unk>", freq=unk_freq)
        self.add_word("<pad>", freq=pad_freq)
        #self.add_word("<url>", freq=url_freq)

        embedding_tensor = torch.zeros(self.word_num, embedding_dim)
        embedding_tensor[-2].uniform_(-0.25, 0.25)
        for word in self.idx2word:
            self.has_word_embedding[word] = 0
        with open(pretrained_source, 'r') as f:
            start_time = time.time()
            for ind, line in enumerate(f):
                try:
                    parseline = line.strip().split()
                    embedding = [float(item) for item in parseline[-embedding_dim:]]
                    word = " ".join(parseline[:-embedding_dim])
                    if word in self.word2idx.keys():
                        embedding_tensor[self.word2idx[word]] = torch.Tensor(embedding)
                        self.has_word_embedding[word] = 1
                    
                except:
                    print(line)

                finally:
                    if (ind + 1) % 10000 == 0:
                        dur_time = time.time() - start_time
                        num_in_dict = sum(self.has_word_embedding.values())
                        print("%s in source, %s / %s in dict, time: %.2f s" % (ind + 1, num_in_dict, self.word_num, dur_time))
                        start_time = time.time()
                
        self.embedding = nn.Embedding.from_pretrained(embedding_tensor)


    def build_dict(self,path,special_token=None):
        """
        :param path:
        :param savepath:
        :param EOS:
        :return:
        """
        if special_token is not None:
            for st in special_token:
                self.add_word(st,freq=100)
                # self.idx2freq(100)

        with open(path,'r') as file:
            lines=file.readlines()
            for line in lines:
                line=line.strip().split()
                for word in line:
                    self.add_word(word)
    def build_dict_on_list(self,lines,special_token=None):
        """
        :param path:
        :param savepath:
        :param EOS:
        :return:
        """
        if special_token is not None:
            for st in special_token:
                self.add_word(st,freq=100)
                # self.idx2freq(100)


        for line in lines:
            line=line.strip().split()
            for word in line:
                self.add_word(word)

    def save_dict(self,savepath):
        with open(savepath+'word_dict.pkl','wb') as file:
            pkl.dump(self,file)

    def set_threshold(self,threshold):
        self.threshold=threshold
        realdict={}
        realidx=[]
        realfq=[]
        for i, fq in enumerate(self.idx2freq):
            if fq >= self.threshold:
                cword=self.idx2word[i]
                realdict[cword]=len(realidx)
                realidx.append(cword)
                realfq.append(fq)
        self.word2idx=realdict
        self.idx2freq=realfq
        self.idx2word=realidx
        self.word_num = len(self.idx2word)


    def texts_to_chunked_len_seq(self,texts,chunked_len):
        word_list=[[self.word2idx[word] if word in self.word2idx else self.word2idx['<unk>'] for word in sentence.split()[:chunked_len]] for sentence in texts]
        word_list=[sentence+[self.word2idx['<pad>']]*(chunked_len-len(sentence)) if len(sentence)<chunked_len else sentence for sentence in word_list]
        return word_list

    def print_statistics(self):
        print("threshold:{th},vocabsize:{vb}".format(th=self.threshold,vb=self.word_num))

    def seqs_to_texts(self,seqs):
        texts=[]
        for seq in seqs:
            texts.append([self.idx2word[i] for i in seq])
        return texts


def save_object(obj,path):
    with open(path,'wb') as files:
        pkl.dump(obj,files)


def load_object(path):
    with open(path,'rb') as files:
        obj = pkl.load(files)
    return obj


def split_data(data,label,ratio):
    """

    :param data:
    :param lable:
    :param ratio: an int
    :return:
    """

    length=len(data)
    pivot=(ratio-1)*length//ratio
    train_data,train_label=data[:pivot],label[:pivot]
    valid_data,valid_label=data[pivot:],label[pivot:]
    return train_data,train_label,valid_data,valid_label



class MyDataset(Dataset):
    def __init__(self,data_path, label_path=None, feature_path=None, word_dict=None,max_len=30):
        """
        :param path:  .pkl files (cleaned sentences for training
        :param word_dict:  a list loaded from  word_dict.pkl
        :max_len:  the max len of a sentence , any sentence longer than this will be chunked
        """
        self.label_path = label_path
        self.feature_path = feature_path
        with open(data_path,'rb') as file:
            sentences=pkl.load(file)
        if label_path:
            with open(label_path,'rb') as file2:
                self.labels=pkl.load(file2)
            self.labels=torch.tensor(self.labels)
        if feature_path:
            with open(feature_path, 'rb') as f:
                self.features = pkl.load(f)
            self.features = torch.FloatTensor(self.features)
        self.seqs=word_dict.texts_to_chunked_len_seq(sentences, max_len)
        self.seqs=torch.tensor(self.seqs)

        if torch.cuda.is_available():
            self.seqs=self.seqs.cuda()
            if label_path:
                self.labels=self.labels.cuda()
            if feature_path:
                self.features = self.features.cuda()


    def __getitem__(self, item):
        if self.feature_path:
            if self.label_path:
                return self.seqs[item],self.labels[item], self.features[item]
            else:
                return self.seqs[item], self.features[item]
        else:
            if self.label_path:
                return self.seqs[item], self.features[item]
            else:
                return self.seqs[item]

    def __len__(self):
        return len(self.seqs)



if __name__ == '__main__':
    qids, labels, questions = read_in_file("data/test.csv", labeled=False)
    cleaned_test = clean_sentences(questions)
    save_object(cleaned_test, "data/test_data.pkl")

    qids,labels,questions=read_in_file("data/train.csv")
    cleaned=clean_sentences(questions)
    train_data, train_label, valid_data, valid_label=split_data(cleaned,labels,8)

    if not os.path.exists('data'):
        os.makedirs('data')

    save_object(cleaned,'data/cleaned_questions.pkl')
    save_object(qids,'data/qids.pkl')
    save_object(labels,'data/labels.pkl')
    save_object(train_data,'data/train_data.pkl')
    save_object(train_label,'data/train_label.pkl')
    save_object(valid_data,'data/valid_data.pkl')
    save_object(valid_label,'data/valid_label.pkl')


    print(cleaned[3:5])
    word_dict=Worddict()
    word_dict.build_dict_on_list(cleaned,special_token=['<unk>','<pad>'])
    for i in range(len(word_dict.idx2word)):
        word = word_dict.idx2word[i]
        if word.find('<n>') >= 0 or word.find('<url>') >= 0 or word.find('<math>') >= 0 or word.find('<time>') >= 0:
            print('{}, freq: {}'.format(word, word_dict.idx2freq[i]))

    word_dict.set_threshold(3)
    word_dict.print_statistics()
    seq=word_dict.texts_to_chunked_len_seq(cleaned[3:5],20)

    word_dict.save_dict('data/')
    print(seq)
    print(word_dict.seqs_to_texts(seq))

    word_dict = load_object('./data/word_dict.pkl')
    word_dict.build_embedding_from_pretrained("../embeddings/glove.840B.300d/glove.840B.300d.txt")
    word_dict.save_dict('data/')
    for word in word_dict.idx2word:
        if not word_dict.has_word_embedding[word]:
            print("<s>", word, "<e>")
    #while True:
    #    word = input("input:")
    #    if word in word_dict.word2idx.keys():
    #        wordid = word_dict.word2idx[word]
    #        print(wordid)
    #        print(word_dict.embedding(torch.LongTensor([wordid])).shape)
    #    else:
    #        print("not exist")






