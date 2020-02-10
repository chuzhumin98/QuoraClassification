import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch.utils.data import *
from bk_quora_main import args
import math
import matplotlib.pyplot as plt
from torch.autograd import Variable

class ScaledDotProductAttention(nn.Module):
    """compute output of a single attention with given Q,K,V
        Attributes:
            dropout: do dropout-operation to the softmax score between Q,K
            softmax: softmax layer to compute the score of Q,K
            att: softmax(QK/sqrt(dim))
            score: matching score for Q,K
    """

    def __init__(self,dropoutrate=0.1,maskhalf=False):
        """Initiate the layer
        :param dropoutrate: the probability to mask neuron with zeros
        """
        self.maskhalf=maskhalf
        super(ScaledDotProductAttention,self).__init__()
        self.dropout=nn.Dropout(dropoutrate)
        self.softmax=nn.Softmax(dim=-1)
        #self.mask=None


    def forward(self,Q,K,V,):
        """the attention layer's forward pass
        :param Q: query
        :param K: key
        :param V: value
        :return: softmax(QK/sqrt(dim))V , score
        """
        # print("Q",Q)

        att=torch.bmm(Q,K.transpose(1,2))
        # print(self.att)
        if self.maskhalf==True:
            # print('mask..')
            self.mask = subsequent_mask(att.shape[-1])
            att=att.masked_fill(self.mask==0,-1e9)
            # print(self.att)
            #print(self.att[0,:,:])
        # print(self.att)
        dim=Q.size()[-1]
        att=att / np.sqrt(dim)
        #print(self.att)
        att=self.softmax(att)  ##不会数值爆炸吗
        att = self.dropout(att)
        #print(self.score)
        # print(V)
        att=torch.bmm(att,V)
        return att

def subsequent_mask(size,cuda=True):
    "Mask out subsequent positions."
    attn_shape = (1,size, size)
    subsequent_mask = np.triu(np.ones(attn_shape,dtype=np.float32), k=1).astype('uint8')
    #print(torch.from_numpy(subsequent_mask))
    a = torch.from_numpy(subsequent_mask) == 0
    if cuda:
        a=a.cuda(args.gpuidx)
    return a


class MultiheadAttention(nn.Module):
    """multihead Attention
    """
    def __init__(self,modeldim,headnum=8,dropoutrate=0.3,maskhalf=False):
        super(MultiheadAttention, self).__init__()
        self.maskhalf=maskhalf
        self.h=headnum
        try:
            modeldim % headnum!=0
        except ValueError:
            print("modeldim should be a multiply of headnum")
        self.dimperhead=modeldim//headnum
        self.Qlinear=nn.Linear(modeldim,modeldim)
        #print(self.Qlinear.weight.data)
        self.Klinear=nn.Linear(modeldim,modeldim)
        self.Vlinear=nn.Linear(modeldim,modeldim)
        self.dropout=nn.Dropout(dropoutrate,inplace=True)
        self.attention=ScaledDotProductAttention(dropoutrate=dropoutrate,maskhalf=maskhalf)
        self.finallinear=nn.Linear(modeldim,modeldim)
        self.connect=ResandNorm(modeldim)

    def forward(self,Q,K,V):
        """
        :param Q:
        :param K:
        :param V:
        :return:
        """
        batch_size=Q.size()[0]
        Q_cat=self.Qlinear(Q)
        K_cat=self.Klinear(K)
        V_cat=self.Vlinear(V)
        Q_reshape=Q_cat.view(batch_size*self.h,-1,self.dimperhead)
        K_reshape = K_cat.view(batch_size * self.h, -1, self.dimperhead)
        V_reshape = V_cat.view(batch_size * self.h, -1, self.dimperhead)
        # #
        # Q_reshape=Q.view(batch_size*self.h,-1,self.dimperhead)
        # K_reshape = K.view(batch_size * self.h, -1, self.dimperhead)
        # V_reshape = V.view(batch_size * self.h, -1, self.dimperhead)
        # Q_reshape = Q_cat.view(batch_size ,-1,self.h, self.dimperhead).transpose(1,2).contiguous().view(batch_size*self.h,-1,self.dimperhead)
        # K_reshape = K_cat.view(batch_size ,-1,self.h, self.dimperhead).transpose(1,2).contiguous().view(batch_size*self.h,-1,self.dimperhead)
        # V_reshape = V_cat.view(batch_size ,-1,self.h, self.dimperhead).transpose(1,2).contiguous().view(batch_size*self.h,-1,self.dimperhead)
        att =self.attention(Q_reshape,K_reshape,V_reshape)
        att=att.view(batch_size,-1,self.dimperhead*self.h)
        att=self.finallinear(att)
        output=self.dropout(att)
        output=self.connect(output,V)
        return output

class ResandNorm(nn.Module):
    def __init__(self,modeldim):
        super(ResandNorm,self).__init__()
        self.modeldim=modeldim
        self.layernorm=nn.LayerNorm(modeldim)

    def forward(self,input,res):
        """

        :param input: input must be the direct last layer
        :param res: input of former layer as residual
        :return:
        """
        #input/np.sqrt(self.modeldim)
        return self.layernorm(input+res)

class Feedforward(nn.Module):
    def __init__(self,modeldim,dff,dropoutrate=0.3):
        super(Feedforward,self).__init__()
        self.w1=nn.Linear(modeldim,dff)
        self.w2=nn.Linear(dff,modeldim)
        self.dropout=nn.Dropout(dropoutrate)
        self.connect=ResandNorm(modeldim)

    def forward(self, input):
        """ compute  connect_with_layer_norm(max(0,W1*input+b)*W+b,input)
        :param input:
        :return:
        """

        output=self.w2(self.dropout(F.relu(self.w1(input),inplace=True)))
        return self.connect(output,input)

class Encoderlayer(nn.Module):
    """ The total setup for Encoder layer
        Attributes:
        multiheadatt: multiheadattion layer
        feedforward: feedforward layer
    """
    def __init__(self,modeldim,dff,headnum=8,dropoutrate=0.3,maskhalf=False):
        super(Encoderlayer,self).__init__()
        self.multiheadatt=MultiheadAttention(modeldim=modeldim,headnum=headnum,dropoutrate=dropoutrate,maskhalf=maskhalf)

        self.feedforward=Feedforward(modeldim,dff,dropoutrate)

    def forward(self,input):
        """

        :param input: input the word embedding mixed with position embedding
        :return: the output of the encoder .It can be used as the input of the next layer
        """
        #print("---",input.size())
        #mask=subsequent_mask(input.shape[1])
        # print(input.size())

        attention_out=self.multiheadatt(input,input,input)#,mask)
        feedforward_out=self.feedforward(attention_out)
        return feedforward_out






class PositionEncoding(nn.Module):
    def __init__(self,modeldim,dropout=0.3,maxlen=5000,base=10000.0):
        super(PositionEncoding,self).__init__()
        self.dropout=nn.Dropout(dropout,inplace=True)
        pe = torch.zeros(maxlen, modeldim)
        L=torch.arange(0,modeldim,2).float()
        divterm = torch.exp((torch.arange(0, modeldim, 2).float() *
                             -(math.log(10000.0) / modeldim)))
        #print(divterm)
        # pos=np.array([i for i in range(maxlen)])
        position=torch.arange(0,maxlen).unsqueeze(1).float()
        #print(position)


        pe[:,0::2] = torch.sin(position*divterm)
        pe[:,1::2] = torch.cos(position*divterm)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self,input_len):
        # max=torch.max(input_len)
        x =Variable(self.pe[:,:input_len],
                         requires_grad=False)
        #print(self.dropout(x).size())

        return self.dropout(x)

class Embeddings(nn.Module):
    def __init__(self, modeldim, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, modeldim,padding_idx=0)
        self.modeldim = modeldim

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.modeldim)




def positionmask(seq_q,seq_k):  ##seq better is tensor
    """

    :param seq_q: token of Q
    :param seq_k: token of K
    :return:
    """
    if type(seq_q)==np.ndarray:
        seq_q=torch.from_numpy(seq_q)
        seq_k=torch.from_numpy(seq_k)
    elif type(seq_q)==torch.Tensor:
        pass

    len_q=seq_q.size(1)
    pad_=seq_k.eq(0)
    pad=pad_.unsqueeze(1).expand(-1,len_q,-1)
    return pad

class Encoder(nn.Module):
    """Several Layers stacked together to form a transformer encoder

    """
    def __init__(self,
                 num_layers=6,
                 modeldim=512,
                 dff=2048,
                 headnum=8,
                 dprate=0.3,
                 maskhalf=False
                 ):
        super(Encoder,self).__init__()
        print("using {a} layers".format(a=num_layers))
        self.maskhalf=maskhalf
        self.encoderlayers=nn.ModuleList([Encoderlayer(modeldim=modeldim, dff=dff,headnum=headnum,
                                                       dropoutrate=dprate,maskhalf=maskhalf) for _ in range(num_layers)])
        self.positionembedding=PositionEncoding(modeldim=modeldim)



    def forward(self,input): ##seq_len is a ndarray?\
        seq_len=input.size()[1]
        # print("max is ",torch.max(seq),"vocabsize is",self.vocab_size)
        output=self.positionembedding(seq_len)+input
        # self_attention_mask=positionmask(seq,seq)  #TODO:possible # is wrong
        for layer in self.encoderlayers:
            output=layer(output)
            # print(output)
        return output

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()

    def forward(self):
        return

class Transformer(nn.Module):
    def __init__(self,middle_dim,num_layer=6,modeldim=512,dff=2048,headnum=6,dprate=0.2,maskhalf=False):
        super(Transformer,self).__init__()
        self.name='transformer'
        self.maskhalf=maskhalf
        self.encoder=Encoder(num_layer,modeldim,dff,headnum,dprate,maskhalf)
        self.linear=nn.Linear(modeldim,middle_dim)
        self.linear2=nn.Linear(middle_dim,2)

    def forward(self,input,feature):
        """
        :param input: a tensor (batchsize,maxlen,modeldim)
        :param input_len: a ndarray(batchsize,)
        :return: a tensor(batchsize, maxlen ,vocabsize)
        """
        assert type(input)==torch.Tensor
        output=self.encoder(input)
        output=torch.mean(output,dim=1)
        output=self.linear(output)
        output=self.linear2(output)
        return output

def crossentropy_mask(target):
    """
    :param target: target sentence with 0 indicates padding
    :return: float mask matrix tensor
    """
    pad=1-target.eq(0)
    return pad.float()


class Criterion(nn.Module):
    """
    """
    def __init__(self):
        super(Criterion,self).__init__()
        self.losslayer=nn.CrossEntropyLoss()

    def forward(self,input,target):
        """
        :param input: a tensor (batchsize,maxlen,vocabsize)
        :param target: a tensor (batchsize,maxlen)
        :param target_len: a ndarray (batchsize,)
        :return: loss,ppl both scaler
        """
        input=torch.transpose(input,1,2)
        mask = crossentropy_mask(target)
        loss=self.losslayer(input,target) ##此处loss报表(neg)过
        #print("size",self.loss.size())
        #print("loss,mask:",self.loss,mask,torch.tensor(target_len))
        # loss=loss*mask
        # loss=loss.sum(dim=1)/torch.tensor(target_len).float()
        # loss=loss.mean()
        return loss
#
# def paddingtotensor(input,maxlen):
#     """
#     :param input: ndarray in which each element is a list
#     :param maxlen: the max len of this batch's seq,int value
#     :return: padded torch tensor (batchsize,maxlen)
#     """
#     output=torch.from_numpy(np.array([i+[0,]*(maxlen-len(i)) for i in input]))
#     return output

# def get_seq_len(input):
#     """
#     :param input: ndarray in which each element is a list
#     :return: ndarray of (batchsize,)
#     """
#     return np.array([len(i) for i in input])

def main():
    plt.figure(figsize=(15, 5))
    pe = PositionEncoding(20,0)
    y = pe.forward(torch.from_numpy(np.array([100,100,100,100])))
    plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    print("draw")
    plt.show()



class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        #print(self._step)
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_lr(self):
        return self._rate

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))






if __name__=="__main__":
    main()













