import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

MAX=-1e9

class TestModelSD(nn.Module):
    def __init__(self, args, name="TestModel"):
        super().__init__()
        self.args = args
        self.name = name

        self.lstm = nn.LSTM(
                input_size=args.embedding_dim,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
                bidirectional=args.bidirectional
                )
        self.drop = nn.Dropout(args.dropout)
        self.fc = nn.Linear(args.hidden_size, args.label_nums)
        self.lt=nn.Linear(2*self.args.hidden_size,self.args.hidden_size)
        self.lt2=nn.Linear(self.args.hidden_size,self.args.hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, inputs,padding):
        inputs = torch.transpose(inputs, 0, 1)
        x, h = self.lstm(inputs)
        h=h[0]
        # print(x.size(),h.size())
        x = self.tanh(self.lt2(x))
        h = h.transpose(0, 1)
        h = h.reshape([h.size()[0], -1])
        # print(h.size())
        h = self.lt(h)
        x = x.transpose(0, 1)

        self.att = attention(h, x,padding).unsqueeze(2)
        x = x.mul(self.att)
        x = torch.sum(x, dim=1)
        output = x.view(-1, inputs.size(1), self.args.hidden_size)
        # print(output.size())
        outputs = torch.mean(output, 0)
        outputs = F.relu(self.fc(outputs))
        return outputs


def attention(h,V,padding_idx):
    padding_idx=padding_idx.float()
    # print(h.size(),V.size())
    att=torch.matmul(h.unsqueeze(1),V.transpose(1,2)).squeeze(1)
    att=att*(1-padding_idx)+padding_idx*MAX
    # print(att)
    att=F.softmax(att,dim=1)
    # print(att)
    return att

class TestModel(nn.Module):
    def __init__(self, args, name="TestModel"):
        super().__init__()
        self.args = args
        self.name = name

        self.lstm = nn.LSTM(
                input_size=args.embedding_dim,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
                bidirectional=args.bidirectional
                )
        self.drop = nn.Dropout(args.dropout)
        self.featurefc = nn.Linear(100, args.feature_size)
        self.fc = nn.Linear(args.hidden_size + args.feature_size, args.label_nums)

    def forward(self, inputs, feature):
        inputs = torch.transpose(inputs, 0, 1)
        outputs, _ = self.lstm(inputs)
        output = outputs.view(-1, inputs.size(1), self.args.hidden_size)
        outputs = torch.mean(outputs, 0)
        zip_feature = self.featurefc(feature) 
        outputs = torch.cat((outputs, zip_feature), 1)
        outputs = F.relu(self.fc(self.drop(outputs)))
        return outputs

class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        key_T = key.transpose(-2, -1)
        weights = F.softmax(torch.matmul(query, key_T) / np.sqrt(key_T.size(-2)), 1)
        outputs = torch.matmul(weights, value)
        return outputs

class DotProductAttentioninBatches(nn.Module):
    """Key should be in shape (seq_num, batch_size, embedding_dim)"""
    def __init__(self):
        super().__init__()
        self.attn = DotProductAttention()
        self.output = torch.Tensor()

    def to(self, device):
        super().to(device)
        self.output = self.output.to(device)

    def forward(self, query, key, value):
        zero = self.output
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        for q, k, v in list(zip(query, key, value)):
            q = q.view(1, q.size(0))
            output = self.attn(q, k, v)
            self.output = torch.cat((self.output, output), 0)
        output = self.output.view(value.size(0), value.size(2))
        self.output = zero
        return output


class DotProductSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = DotProductAttention()

    def forward(self, inputs):
        return self.attn(inputs, inputs, inputs)

class MultiheadSelfAttention(nn.Module):
    def __init__(self, num_head, d_model, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.num_head = num_head
        self.d_model = d_model
        self.LinQ = nn.Linear(d_model, d_model // num_head)
        self.LinK = nn.Linear(d_model, d_model // num_head)
        self.LinV = nn.Linear(d_model, d_model // num_head)
        self.LinO = nn.Linear(d_model, d_model)
        self.attn = DotProductAttentioninBatches()

    def forward(self, inputs):
        heads = tuple(self.attn(self.LinQ(self.drop(inputs)), self.LinK(self.drop(inputs)), self.LinV(self.drop(inputs))) for _ in range(self.num_head))
        concat = torch.cat(heads, -1)
        return self.LinO(concat)

class QuoraModel(nn.Module):
    def __init__(self, args, name="QuoraModel"):
        super().__init__()
        self.args = args
        self.name = name

        if args.model == "lstm":
            self.rnn = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional)
        elif args.model == "gru":
            self.rnn = nn.GRU(input_size=args.embedding_dim, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional)
        self.drop = nn.Dropout(args.dropout)
        self.attn = DotProductAttentioninBatches()
        self.featurefc = nn.Linear(100, args.feature_size)
        self.fc = nn.Linear(args.hidden_size + args.feature_size, 2)

        self.init_weights()

    def to(self, device):
        super().to(device)
        self.attn.to(device)


    def init_weights(self):
        nn.init.xavier_uniform_(self.featurefc.weight) 
        nn.init.xavier_uniform_(self.fc.weight) 

    def forward(self, inputs, feature):
        outputs = torch.transpose(inputs, 0, 1).contiguous()
        outputs, _ = self.rnn(outputs)
        if self.args.bidirectional:
            outputs = outputs.view(-1, inputs.size(0), 2, self.args.hidden_size)
            outputs = torch.mean(outputs, 2)
            last_state = (outputs[-1] + outputs[0]) / 2
        else:
            outputs = outputs.view(-1, inputs.size(0), self.args.hidden_size)
            last_state = outputs[-1]
        if self.args.attention:
            output = self.attn(last_state, outputs, outputs)
        else:
            output = torch.mean(outputs, 0)
        zip_feature = F.relu(self.featurefc(feature)) 
        outputs = torch.cat((output, zip_feature), 1)
        outputs = F.relu(self.fc(self.drop(outputs)))
        return outputs



class LSTMMODEL(nn.Module):
    def __init__(self,args,label_num=5):
        super(LSTMMODEL,self).__init__();
        self.num_vocab=args.num_vocab
        self.embedding_dim=args.embedding_dim
        self.hidden_size=args.hidden_size
        self.stack_lstm=args.stack_lstm
        self.dp=args.drop_out
        self.attention=args.attention
        self.bilstm=args.bilstm
        self.mid_size=args.mid_size
        self.label_num=label_num
        self.embedding=nn.Embedding(self.num_vocab,self.embedding_dim)
        self.lstm=nn.LSTM(input_size=self.embedding_dim,
                          hidden_size=self.hidden_size,
                          num_layers=self.stack_lstm,
                          dropout=self.dp,
                          bidirectional=self.bilstm)
        if self.bilstm==True and self.attention==True:
            self.lt = nn.Linear(2 *self.stack_lstm* self.hidden_size, self.hidden_size)
            self.lt2=nn.Linear(2*self.hidden_size,self.hidden_size)
            self.fc1=nn.Linear(self.hidden_size,self.mid_size)


        elif self.bilstm==True and self.attention==False:
            self.fc1=nn.Linear(2*self.stack_lstm*self.hidden_size,self.mid_size)
        elif self.bilstm==False and self.attention==True:
            self.lt = nn.Linear(self.stack_lstm*self.hidden_size, self.hidden_size)
            self.fc1=nn.Linear(self.hidden_size,self.mid_sizem)
            self.lt2 = nn.Linear(self.hidden_size, self.hidden_size)
        else:
            self.fc1=nn.Linear(self.stack_lstm*self.hidden_size,self.mid_size)


        self.fc2=nn.Linear(self.mid_size,self.label_num)
        self.dropout=nn.Dropout(self.dp,inplace=True)
        self.tanh=nn.Tanh()


    def forward(self, input):

        x=self.embedding(input)
        if self.attention==True:
            x=self.lstm(x)
            x,h=x[0],x[1][0]
            x=self.tanh(self.lt2(x))
            h=h.transpose(0,1)
            h=h.reshape([h.size()[0],-1])
            h=self.lt(h)
            x=x.transpose(0,1)
            att=attention(h,x).unsqueeze(2)
            x =x.mul(att)
            x=torch.sum(x,dim=1)
        else:
            x=self.lstm(x)[1][0]
            x=x.transpose(0,1)
            x=x.reshape([x.size()[0],-1])
        x=self.fc1(x)
        x=F.relu(x,inplace=True)
        x = self.dropout(x)
        x=self.fc2(x)
        return x

def attention(h,V):
    att=torch.matmul(h.unsqueeze(1),V.transpose(1,2)).squeeze(1)
    att=F.softmax(att,dim=1)
    return att


def self_attention(V):
    Q=V
    K=V.transpose(1,2)
    att=torch.matmul(Q,K)
    att=F.softmax(att,dim=2)
    return att
