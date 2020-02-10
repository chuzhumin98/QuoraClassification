import pickle as pkl
from dataloader import Worddict
import seaborn as sea
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
def attention_study():
    with open("./data/attention.pkl",'rb') as file1:
        attention = pkl.load(file1)

    with open("./data/valid_data.pkl",'rb') as file2:
        valid_data=pkl.load(file2)
        # print(valid_data[0])

    with open("./data/valid_label.pkl",'rb') as file3:
        valid_label=pkl.load(file3)

    with open("./data/word_dict.pkl",'rb') as file4:
        word_dict=pkl.load(file4)

    with open("./data/ans.pkl",'rb') as file5:
        ans = pkl.load(file5)
        print(len(ans),len(valid_label))
    # valid_sentence=word_dict.seqs_to_texts(valid_data)
    head=[]
    tail=[]
    idx11=[]
    idx01=[]
    idx10 = []
    idx00 = []
    # if 0 == 1:
    #     print('恭喜，你猜对了。')  # 新块从这里开始
    #     print('(但你没有获得任何奖品！)')  # 新块在这里结束
    # elif 0 < 1:
    #     print('不对，你猜的有点儿小')  # 另一个块
    # else:
    #     print('不对，你猜的有点大')
    # # print('完成')

    for i in range(len(valid_data)):

        if (valid_label[i]==1):
            if(ans[i]==1):
            # print(valid_label[i],ans[i])
                idx11.append(i)
            else:
                idx10.append(i)
        else:
            # print(valid_label[i], ans[i])
            if (ans[i] == 1):
                # print(valid_label[i],ans[i])
                idx01.append(i)
            else:
                idx00.append(i)
        head.append(attention[0,0])
        att=attention[i].transpose().reshape([1, -1])
        # print(np.sum(att==0))
        tlen=att.shape[-1]-np.sum(att==0)
        tail.append(attention[0,tlen-1])
    # print(head,'\n',tail)
    print(len(idx11),len(idx10),len(idx01),len(idx00))
    return sum(head)/len(head),sum(tail)/len(tail),idx11,idx10,idx01,idx00


def show_hot(L,ht):
    with open("./data/attention.pkl",'rb') as file1:
        attention = pkl.load(file1)

    with open("./data/valid_data.pkl",'rb') as file2:
        valid_data=pkl.load(file2)
        # print(valid_data[0])

    with open("./data/valid_label.pkl",'rb') as file3:
        valid_label=pkl.load(file3)




    items_num=len(L)



    cmap = sea.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
    for idx,i in enumerate(L):
        # print(valid_data[i],valid_label[i])
        data_list=valid_data[i].split()
        plt.subplot(items_num,1,idx+1)
        att=attention[i].transpose().reshape([1, -1])
        # print(np.sum(att==0))
        tlen=att.shape[-1]-np.sum(att==0)
        # print(tlen)

        att=att[:,:tlen]
        # print(att)
        data_list=data_list[:tlen]
        # print(att,data_list)

        df=pd.DataFrame(att,columns=data_list)
        ax=sea.heatmap(df, linewidths=0.05, vmax=np.max(att),vmin=np.min(att), annot=True,xticklabels="auto",annot_kws={'size':4,'weight':'bold', 'color':'blue','rotation':90},cmap="BuPu")
    # ax = sns.countplot(x="Column", data=ds)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right",size=9)
    plt.tight_layout()
    # sea.heatmap(L)
    plt.show()

    # for idx,i in enumerate(L):
    #     # print(valid_data[i],valid_label[i])
    #     plt.subplot(items_num,1,idx+1)
    #     att=attention[i]
    #     # print(np.sum(att==0))
    #     tlen=len(att)-np.sum(att==0)
    #     # print(tlen)
    #
    #     att=att[:tlen]
    #     # print(att)
    #     plt.plot(att)
    #     # sea.heatmap(att, linewidths=0.05, vmax=np.max(att),vmin=np.min(att), cmap='rainbow')
    # # sea.heatmap(L)
    # plt.show()








if __name__=='__main__':
    head,tail,idx11,idx10,idx01,idx00=attention_study()
    print(head,tail)

    # show_hot([58496,58457],(head,tail))

    # show_hot(idx00[2:5],(head,tail))

    show_hot(idx01[5:9],(head,tail))
    print(idx01[5:9])
    show_hot(idx10[17:19],(head,tail))
    print(idx10[17:19])
    show_hot(idx11[17:20],(head,tail))
    print(idx11[17:20])

    L=[80,2580,101]
    show_hot(L, (head, tail))




