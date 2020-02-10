import imp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class Vocabulary(object):
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.num_words = 2

    def __len__(self):
        return self.num_words

    def addSentence(self, sentence):
        for word in sentence.split():
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))
        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.num_words = 0  
        for word in keep_words:
            self.addWord(word)

def GetEmbeddings(filepath, name="pretrained_embedding"):
    voc = Vocabulary(name)
    embeddings = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                parseline = line.split()
                word = parseline[0]
                embedding = [eval(item) for item in parseline[1:]]
                voc.addWord(word)
                embeddings.append(embedding)
            except:
                print(line)
    return voc, nn.Embedding.from_pretrained(torch.Tensor(embeddings))
    

    

def main():
    voc, embeddings = GetEmbeddings("../embeddings/glove.840B.300d/glove.840B.300d.txt")
    print(embeddings[0])

if __name__ == "__main__":
    main()
