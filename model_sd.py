import torch.nn as nn
import torch.nn.functional as F
import torch

# class TestModel(nn.Module):
#     def __init__(self, args, name="TestModel"):
#         super().__init__()
#         self.args = args
#         self.name = name
#
#         self.lstm = nn.LSTM(
#                 input_size=args.embedding_dim,
#                 hidden_size=args.hidden_size,
#                 num_layers=args.num_layers,
#                 dropout=args.dropout,
#                 bidirectional=args.bidirectional
#                 )
#         self.drop = nn.Dropout(args.dropout)
#         self.fc = nn.Linear(args.hidden_size, args.label_nums)
#
#     def forward(self, inputs):
#         inputs = torch.transpose(inputs, 0, 1)
#         outputs, _ = self.lstm(inputs)
#         output = outputs.view(-1, inputs.size(1), self.args.hidden_size)
#         outputs = torch.mean(outputs, 0)
#         outputs = F.relu(self.fc(outputs))
#
#         return outputs

MAX=-1e9

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

        self.uw=torch.randn((self.hidden_size,1),requires_grad=True)



    def forward(self, x):
        print(x)

        # x=self.embedding(input)
        if self.attention==True:
            x=self.lstm(x)
            x,h=x[0],x[1][0]
            x=self.tanh(self.lt2(x))
            # h=h.transpose(0,1)
            # h=h.reshape([h.size()[0],-1])
            # h=self.lt(h)
            x=x.transpose(0,1)

            self.att=attention(self.uw,x).unsqueeze(2)
            x =x.mul(self.att)
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




def self_attention(V):
    Q=V
    K=V.transpose(1,2)
    att=torch.matmul(Q,K)
    att=F.softmax(att,dim=2)
    return att
