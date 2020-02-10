import csv
import time

import dataloader
from dataloader import Worddict
import model

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import json
import trfencoder
import pickle as pkl

E = 1e-8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#torch.distributed.init_process_group(backend="nccl", world_size=4, rank=0)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=8, help="random seed DEFAULT: 8")
parser.add_argument('--max_len', type=int, default=30, help="max length DEFAULT: 30")
parser.add_argument('--batch_size', type=int, default=32, help="batch size DEFAULT: 32")
parser.add_argument('--hidden_size', type=int, default=300, help="hidden size DEFAULT: 300")
parser.add_argument('--feature_size', type=int, default=50, help="feature size DEFUALT: 50")
parser.add_argument('--num_layers', type=int, default=3, help="LSTM layer number DEFAULT: 2")
parser.add_argument('--disp_freq', type=int, default=100, help="display frequency DEFAULT: 100")
parser.add_argument('--num_epochs', type=int, default=3, help="number of epochs DEFAULT: 10")
parser.add_argument('--dropout', type=float, default=0.4, help="dropout DEFAULT: 0.4")
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate DEFAULT: 3e-4")
parser.add_argument("--dcstep",default=3000,type=int)

parser.add_argument('--lr_decay', type=float, default=0.9, help="learning rate decay DEFAULT: 1.0")
parser.add_argument('--sample_weight', type=float, default=0.25, help="sample_weight DEFAULT: 0.25")
parser.add_argument('--save', type=str, default="", help="save model name")
parser.add_argument('--visdom', action="store_true", default=False, help="enable visdom")
parser.add_argument('--bidirectional', action="store_true", default=False, help="LSTM bidirectional")
parser.add_argument('--attention', action="store_true", default=False, help="LSTM with attention")
pargs = parser.parse_args()
if pargs.visdom:
    import visdom
    
    vis = visdom.Visdom()
    assert vis.check_connection()

class args(object):
    max_len = pargs.max_len
    batch_size = pargs.batch_size
    num_vocab = 50000
    embedding_dim = 300
    hidden_size = pargs.hidden_size
    feature_size = pargs.feature_size
    num_layers = pargs.num_layers
    dropout = pargs.dropout
    bidirectional = pargs.bidirectional
    attention = pargs.attention
    lr = pargs.lr
    disp_freq = pargs.disp_freq
    sample_weight = pargs.sample_weight
    label_nums = 2
    num_epochs = pargs.num_epochs
    lr_decay = pargs.lr_decay
    seed = pargs.seed
    name = pargs.save
    gpuidx=5
    dcstep=pargs.dcstep



def accuracy(outputs, labels):
    total = 0
    correct = 0
    _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    return correct / total

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

def train(quora_model, args, word_dict):
    args_dict = dict((name, getattr(args(), name)) for name in dir(args()) if not name.startswith("__"))
    train_set = dataloader.MyDataset('data/train_data.pkl','data/train_label.pkl',feature_path="./data/train_words_attributes_100.pkl", word_dict=word_dict,max_len=args.max_len)
    valid_set = dataloader.MyDataset('data/valid_data.pkl','data/valid_label.pkl','./data/valid_words_attributes_100.pkl', word_dict=word_dict,max_len=args.max_len)
    test_set = dataloader.MyDataset('data/test_data.pkl', None, feature_path='./data/test_words_attributes_100.pkl', word_dict=word_dict, max_len=args.max_len)
    valid_iter = DataLoader(valid_set,batch_size=args.batch_size,shuffle=False)
    test_iter = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    bias = train_set.labels.sum().item() / len(train_set.labels)
    with open('data/train_label.pkl', 'rb') as file:
        train_labels = pkl.load(file)

    rate_0vs1 = float(len(train_labels) - sum(train_labels)) / float(sum(train_labels))
    rate_0vs1 *= args.sample_weight
    weights = [rate_0vs1 if label == 1 else 1 for label in train_labels]
    from torch.utils.data.sampler import WeightedRandomSampler
    sampler = WeightedRandomSampler(weights, num_samples=len(train_labels), replacement=True)

    train_iter=DataLoader(train_set,batch_size=args.batch_size,shuffle=False, sampler=sampler)
    
    len_train_iter = len(train_iter)
    embeddings = word_dict.embedding
    optimizer = optim.Adam(quora_model.parameters(), lr=args.lr)
    # optimizer=trfencoder.NoamOpt(args.embedding_dim, 1, args.dcstep,
    #                             torch.optim.Adam(quora_model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=args.lr_decay)
    criterion = nn.CrossEntropyLoss()
    lossn = 0
    start_time = time.time()
    best_score = 0.0
    for epoch in range(args.num_epochs):
        running_loss = 0
        quora_model.train()
        for i, (data, label, feature) in enumerate(train_iter):
            data, label, feature = data.to(device), label.to(device), feature.to(device)
#            print(data.size(),label.size(),feature.size(),"="*20+"data\n",data,"="*20+'label\n',label,"="*20+"feature\n",feature)
            data = embeddings(data)
            quora_model.zero_grad()

            outputs = quora_model(data, feature)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            f1 = F1score(outputs, label)
            
            running_loss += loss.item()

            if i % args.disp_freq == 0 and i > 0:
                average_loss = running_loss / args.disp_freq
                lossn += 1
                if pargs.visdom:
                    vis.line(np.array([[average_loss]]), np.array([lossn]), win="loss", update="append")
                print("%.3f training loss: %.3f training F1 score: %.3f" % (i / len_train_iter, average_loss, f1))

                dur_time = time.time() - start_time
                print("%s / %s batches trained, %d batches /s" % (i, len_train_iter, args.disp_freq / dur_time))
                start_time = time.time()

                running_loss = 0

                if i % (args.disp_freq * 10) == 0:
                    quora_model.eval()
                    with torch.no_grad():
                        f1s = []
                        for i, (data, label, feature) in enumerate(valid_iter):
                            data, label, feature = data.to(device), label.to(device), feature.to(device)
                            data = embeddings(data)

                            outputs = quora_model(data, feature)
                            f1 = F1score(outputs, label)
                            f1s.append(f1)
                        val_f1 = sum(f1s) / len(f1s)
                        if pargs.visdom:
                            vis.line(np.array([[val_f1]]), np.array([lossn]), win="f1", update="append")
                        print("EPOCH %s, %s validation F1 score: %.3f, previous best score: %.3f" % (epoch, lossn, val_f1, best_score))
                        start_time = time.time()

                    if val_f1 < best_score:
                        # pass
                        scheduler.step()
                    else:
                        best_score = val_f1
    
                        torch.save({
                            "model_args": args_dict,
                            "model_state_dict": quora_model.state_dict()
                            }, "model/" +"best"+ args.name + "_" + quora_model.name + "_" + str(lossn) +  ".pt")
                    quora_model.train()
                    print("current lr: %s" % scheduler.get_lr())

    # preds=[]
    # for i, (data, feature) in enumerate(test_iter):
    #     data ,feature = data.to(device),feature.to(device)
    #     data = embeddings(data)
    #     outputs = quora_model(data,feature)
    #     pred = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
    #     preds.extend(pred)
    #
    # with open("data/json_ans_trf.json", 'w') as fr:
    #     json.dump(preds, fr)
    #     print("test ans saved")


def test(args):
    with open('data/word_dict.pkl','rb') as file:
        word_dict= pkl.load(file)
    args.num_vocab = word_dict.word_num
    test_set = dataloader.MyDataset('data/test_data.pkl', None, feature_path='./data/test_words_attributes_100.pkl', word_dict=word_dict, max_len=args.max_len)
    test_iter = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    ckpt=torch.load("./model/best_transformer_248.pt")
    # model_args=type('args', (object,), ckpt['model_args'])
    # print(model_args)
    quora_model = trfencoder.Transformer(num_layer=3,middle_dim=10,modeldim=args.embedding_dim,dff=args.hidden_size*4)

    # quora_model = trfencoder.Transformer(model_args)#num_layer=args.num_layers,middle_dim=10,modeldim=args.embedding_dim,dff=args.hidden_size*4)

    quora_model.to(device)
    quora_model.load_state_dict(ckpt["model_state_dict"])
    embeddings = word_dict.embedding
    embeddings.to(device)

    preds=[]






    for i,(data,feature) in enumerate(test_iter):
        data=data.to(device)
        data=embeddings(data)
        outputs=quora_model(data,feature)
        pred=torch.argmax(outputs,dim=1).cpu().numpy().tolist()
        preds.extend(pred)
    print(preds)

    with open("data/json_ans.json", 'w') as fr:
        json.dump(preds, fr)
        print("test ans saved")




















def main():
    torch.manual_seed(args.seed)
    from functools import partial
    import pickle
    pickle.load = partial(pickle.load, encoding="utf8")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="utf8")
    with open('data/word_dict.pkl','rb') as file:
        word_dict= pkl.load(file)
    args.num_vocab = word_dict.word_num

    quora_model = trfencoder.Transformer(num_layer=args.num_layers,middle_dim=10,modeldim=args.embedding_dim,dff=args.hidden_size*4)
    embeddings = word_dict.embedding
    embeddings.to(device)
    quora_model.to(device)
    quora_model.share_memory()

    processes = []
    train(quora_model, args, word_dict)


    print(torch.initial_seed())
    #with torch.no_grad():
    #    submit_ans = torch.LongTensor().to(device)
    #    for data in test_iter:
    #        data = data.to(device)
    #        data = embeddings(data)

    #        outputs = quora_model(data)
    #        _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
    #        submit_ans = torch.cat((submit_ans, predicted), 0)
    #    with open("data/json_ans.json", 'w') as fr:
    #        json.dump(submit_ans.cpu().numpy().tolist(), fr)
            


##### show example of batch data####
    #cnt=0
    #for data, label in train_iter:
    #    for s in data:
    #        for w in s:
    #            print(word_dict.idx2word[w.item()], end=" ")
    #        print("")
    #    cnt+=1
    #    if cnt>2:
    #        break






if __name__=="__main__":
    # main()
    test(args)



