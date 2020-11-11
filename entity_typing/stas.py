#    Author:  a101269
#    Date  :  2020/5/6

from tqdm import tqdm
from glob import glob
import re

def add_label(fr,maxlen):
    fr=open(fr,'r',encoding='utf')
    line=fr.read().split('\n')
    for l in line:
        if len(l)>maxlen:
            maxlen=len(l)
    return maxlen
def max_len(fr):
    fr = open(fr, 'r', encoding='utf')
    line = fr.read().split('\n')

#    Author:  a101269
#    Date  :  2020/4/10
import random
import pathlib
from glob import glob

random.seed(1234)
def number_sents(fr):
    abs_path, file_name = pathlib.Path(fr).cwd(), pathlib.Path(fr).name
    pre='dataset\\data_seg\\'
    file_name=file_name.strip('.txt')
    # cashed_name = "cached_example_" + file_name.strip('sdp_')
    # cached_examples_file = abs_path / self.args.cached_path / cashed_name
    print(abs_path)
    print(file_name)
    fr = open(fr, 'r', encoding="utf")
    lines = fr.read()
    sents = lines.strip().split("\n")
    random.shuffle(sents)
    print(len(sents))
    devname = pre + file_name + '.dev.csv'
    trainname = pre + file_name + '.train.csv'
    dev = open(devname, 'w', encoding='utf')
    train = open(trainname, 'w', encoding='utf')
    if len(sents)<500:
        dev_size=20
    elif len(sents)<2000:
        dev_size=30
    elif len(sents)<10000:
        dev_size=int(0.01*len(sents))
    else:
        dev_size=int(0.05*len(sents))
    for i,sent in enumerate(sents):
        if i<dev_size:
            dev.write(sent+'\n')
        else:
            train.write(sent+'\n')
    fr.close()
    dev.close()
    train.close()
def merge(fr,fw):
    fr = open(fr, 'r', encoding="utf")
    lines = fr.read()
    sents = lines.strip().split("\n")
    print(len(sents))
    for i,sent in enumerate(sents):
        fw.write(sent+'\n')
def else_():
    fr = open('dataset/其他.txt', 'r', encoding="utf")
    fw=open('dataset/其他.txt','w',encoding='utf')
    lines = fr.read()
    sents = lines.strip().split("\n")
    for i,word in enumerate(sents):
        fw.write(word + '\t' + '其他' + '\n')

def get_label():
    fr = open('dataset/data_seg/dev.csv', 'r', encoding='utf')
    lines = fr.read()
    sents = lines.strip().split("\n")
    labels=set()
    for i,word in enumerate(sents):
        word=word.split('\t')[1]
        labels.add(word)
    print(labels)
# location，organization，anim,plant,food ,drug,bill，book，building,clothes,event
if __name__ == '__main__':
    maxlen=0

    # files = glob('dataset\*.txt')
    # for file in files:
    #     print(file)
    #     ml=maxlen(file)
    # print()
    # sumlen=0
    # for file in files:
    #     print(file)
    #     number_sents(file)
    fw=open('dataset/data_seg/mix_train.csv','w',encoding='utf')
    files = glob('dataset/data_seg/*.csv')
    files = ['dataset/data_seg/train.csv','dataset/data_seg/dev.csv']
    for file in files:
        merge(file,fw)

