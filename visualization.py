import matplotlib
import torch
from quora_main import args
from repo.model import TestModel
import pickle as pkl
import torch.optim as optim
import torch.nn as nn
import dataloader
from torch.utils.data import DataLoader
# import torch.nn as nn
import torch.nn.functional as F
import numpy as np
E = 1e-8


def F1score(outputs, labels):
    _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
    pairs = list(zip(predicted, labels))
    correct = sum([1 for t, l in pairs if t.item() == 1 and l.item() == 1])
    if labels.sum().item():
        recall = correct / labels.sum().item()
    else:
        recall = 1
    if predicted.sum().item():
        precision = correct / predicted.sum().item()
    else:
        precision = 1
    return (2 * recall * precision + E) / (recall + precision + E)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




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
        for i,fq in enumerate(self.idx2freq):
            if fq>=self.threshold:
                cword=self.idx2word[i]
                realdict[cword]=len(realidx)
                realidx.append(cword)
                realfq.append(fq)
        self.word2idx=realdict
        self.idx2freq=realfq
        self.idx2word=realidx
        self.vocabsize = len(self.idx2word)


    def texts_to_chunked_len_seq(self,texts,chunked_len):
        word_list=[[self.word2idx[word] if word in self.word2idx else self.word2idx['<unk>'] for word in sentence.split()[:chunked_len]] for sentence in texts]
        word_list=[sentence+[self.word2idx['<pad>']]*(chunked_len-len(sentence)) if len(sentence)<chunked_len else sentence for sentence in word_list]
        return word_list

    def print_statistics(self):
        print("threshold:{th},vocabsize:{vb}".format(th=self.threshold,vb=self.vocabsize))

    def seqs_to_texts(self,seqs):
        texts=[]
        for seq in seqs:
            texts.append([self.idx2word[i] for i in seq])
        return texts


def visualize(model_path):
    quora_model=TestModel(args)

    # with open(model_path,'rb') as file:
    check_point=torch.load(model_path)

    quora_model.load_state_dict(check_point['model_state_dict'])

    with open('data/word_dict.pkl','rb') as file:
        word_dict= pkl.load(file)


    embeddings = word_dict.embedding
    # optimizer = optim.Adam(quora_model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(device)#weight = torch.Tensor([bias, 1 - bias]).to(device))

    embeddings.to(device)
    quora_model.to(device)
    torch.load(model_path)

    valid_set = dataloader.MyDataset('data/valid_data.pkl','data/valid_label.pkl',word_dict=word_dict,max_len=args.max_len)
    valid_iter = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    print(len(valid_set))
    attention=[]
    ans=[]



    with torch.no_grad():
        f1s = []
        for i, (data, label) in enumerate(valid_iter):
            data, label = data.to(device), label.to(device)
            idx=padding_idx(data,62706)
            word=data
            data = embeddings(data)

            outputs = quora_model(data,idx)
            pred=list(np.argmax(outputs.cpu().numpy(),axis=-1))
            ans=ans+pred
            # print(quora_model.att.size())
            attention.append(quora_model.att.squeeze(-1).cpu().data.numpy())
            f1 = F1score(outputs, label)
            f1s.append(f1)
            if i%1000==0:
                print('attention:',attention[i][0,:])
                print('word',word[0,:])

        val_f1 = sum(f1s) / len(f1s)
        print('val_f1:',val_f1)

    attention=np.vstack(attention)

    # print('attention:',attention[0:1,:])
    with open('./data/attention.pkl','wb') as file:
        pkl.dump(attention,file)
    with open('./data/ans.pkl','wb') as file:
        pkl.dump(ans,file)
    print(len(ans))

def padding_idx(x,padding_idx):
    return x==padding_idx

if __name__=="__main__":
    visualize('model/TestModel_65.pt')