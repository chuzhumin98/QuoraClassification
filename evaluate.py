import dataloader
from dataloader import Worddict
import model

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import pickle as pkl
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parseargs(arg=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type=str, nargs="+", help="model type, optionally (test, lstm)")
    parser.add_argument("--model_path", nargs="+", type=str, help="model path")
    parser.add_argument("--batch_size", type=int, default=32, help="eval batch size")
    parser.add_argument("--max_len", type=int, default=30, help="eval max len")
    parser.add_argument("--save", type=str, default="", help="save evaluation")
    return parser.parse_args(arg)

def F1score(outputs, labels):
    E = 1e-8
    _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
    pairs = list(zip(predicted, labels))
    correct = sum([1 for t, l in pairs if t.item() == 1 and l.item() == 1])
    zero_correct = sum([1 for t, l in pairs if t.item() == 0 and l.item() == 0])
    if labels.sum().item():
        recall = correct / labels.sum().item()
    else:
        recall = 1
    if predicted.sum().item():
        precision = correct / predicted.sum().item()
    else:
        precision = 1
    return (2 * recall * precision + E) / (recall + precision + E), np.array([[zero_correct, labels.sum().item() - correct], [predicted.sum().item() - correct, correct]]), predicted

def main(args):
    with open('data/word_dict.pkl','rb') as f:
        word_dict= pkl.load(f)
    embeddings = word_dict.embedding
    embeddings.to(device)
    valid_set = dataloader.MyDataset('data/valid_data.pkl','data/valid_label.pkl', feature_path="./data/valid_words_attributes_100.pkl", word_dict=word_dict,max_len=args.max_len)
    test_set = dataloader.MyDataset('data/test_data.pkl', None, feature_path="./data/test_words_attributes_100.pkl", word_dict=word_dict, max_len=args.max_len)
    valid_iter = DataLoader(valid_set,batch_size=args.batch_size,shuffle=False)
    test_iter = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model_types = arg.model
    model_paths = arg.model_path
    checkpoints = [torch.load(path) for path in model_paths]
    model_args = [type('args', (object, ), c['model_args']) for c in checkpoints]
    for ind, checkpoint in enumerate(checkpoints):
        for key, value in checkpoint["model_args"].items():
            setattr(model_args[ind], key, value)
    eval_models = []
    for i, typ in enumerate(model_types):
        if typ == "test":
            eval_model = model.TestModel(model_args[i])
            eval_model.to(device)
            eval_model.load_state_dict(checkpoints[i]["model_state_dict"])
            eval_models.append(eval_model)
        elif typ == "quo":
            eval_model = model.QuoraModel(model_args[i])
            eval_model.to(device)
            eval_model.load_state_dict(checkpoints[i]["model_state_dict"])
            eval_models.append(eval_model)


    with torch.no_grad():
        #valid set
        f1s = []
        mats = np.array([[0, 0], [0, 0]])
        predicts = torch.tensor([], device=device).long()
        for i, (data, label, feature) in enumerate(valid_iter):
            data, label, feature = data.to(device), label.to(device), feature.to(device)
            data = embeddings(data)
            final_output = torch.tensor([], device=device)
            for eval_model in eval_models:
                eval_model.eval()
                outputs = eval_model(data, feature)
                final_output = torch.cat((final_output, outputs), 0)
            outputs = torch.mean(final_output.view(-1, outputs.size(0), outputs.size(1)), 0)
            f1, mat, predicted = F1score(outputs, label)
            predicts = torch.cat((predicts, predicted), 0)
            f1s.append(f1)
            mats += mat
        val_f1 = sum(f1s) / len(f1s)
        print("validation F1 score: %.3f" % (val_f1))
        print("confusion matrix: \n", mats)
        with open("model/" + args.save + "evaluation.txt", 'w') as f:
            f.write(",".join([str(c) for c in predicts.cpu().tolist()]))
            
        #test set
        submit_ans = torch.tensor([], device=device).long()
        for data, feature in test_iter:
            data, feature = data.to(device), feature.to(device)
            data = embeddings(data)
            final_output = torch.tensor([], device=device)
            for eval_model in eval_models: 
                outputs = eval_model(data, feature)
                final_output = torch.cat((final_output, outputs), 0)
            outputs = torch.mean(final_output.view(-1, outputs.size(0), outputs.size(1)), 0)
            _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
            submit_ans = torch.cat((submit_ans, predicted), 0)
        with open("data/json_ans.json", 'w') as fr:
            json.dump(submit_ans.cpu().numpy().tolist(), fr)
            print("test ans saved")



if __name__ == "__main__":
    arg = parseargs()
    main(arg)

