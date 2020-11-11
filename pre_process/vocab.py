#    Author:  a101269
#    Date  :  2020/3/4

from glob import glob
from collections import Counter
import pathlib
import torch

PAD = '<PAD>'  # 词表文件的开头要有<UNK>等
UNK = '<UNK>'
ROOT = '<ROOT>'
VOCAB_PREFIX = [PAD, UNK, ROOT]

def load_vocab(args):
    pos_vocab=None
    know_vocab=None
    rel_vocab=None
    bio_vocab=None
    dataset=args.dataset_path
    knowledge_vocab_path = pathlib.Path(args.knowledge_vocab_path)
    relation_vocab_path = pathlib.Path(args.relation_vocab_path) #knowledge_vocab_path
    pos_vocab_path = pathlib.Path(args.pos_vocab_path)
    bio_vocab_path= pathlib.Path(args.bio_vocab_path)
    for idx in args.columns:
        if idx ==3:
            pos_vocab=PosVocab(dataset,idx=3,vocab_path=pos_vocab_path)
        if idx==8:
            rel_vocab=RelVocab(dataset,idx=8,vocab_path=relation_vocab_path)
        if idx==9:
            pos_set=pos_vocab._id2unit[3:]
            print(pos_set)
            know_vocab= KnowVocab(dataset,idx=9,vocab_path=knowledge_vocab_path,pos_vocab=pos_set)
        if idx ==5:
            bio_vocab=PosVocab(dataset,idx=5,vocab_path=bio_vocab_path)
    vocabs={'pos':pos_vocab,'know':know_vocab,'rel':rel_vocab,'bio': bio_vocab}
    return vocabs


class BaseVocab:
    """ A base class for common vocabulary operations. Each subclass should at least
    implement its own build_vocab() function."""

    def __init__(self, dataset_path=None,idx=0, cutoff=0,vocab_path=None,pos_vocab=None):
        self.idx=idx
        self.cutoff = cutoff
        self.vocab_path = vocab_path
        self.pos_vocab=pos_vocab
        self.build_vocab(dataset_path)



    def build_vocab(self, dataset_path):
        raise NotImplementedError()

    def unit2id(self, unit):
        if unit in self._unit2id:
            return self._unit2id[unit]
        else:
            # print(unit)
            return self._unit2id[UNK]

    def id2unit(self, id):
        return self._id2unit[id]

    def map(self, units):
        return [self.unit2id(x) for x in units]

    def unmap(self, ids):
        return [self.id2unit(x) for x in ids]

    def __len__(self):
        return len(self._id2unit)


class PosVocab(BaseVocab):
    def __init__(self, data=None,idx=3, cutoff=0,vocab_path=None):
        super().__init__(data, idx=idx,cutoff=cutoff,vocab_path=vocab_path)

    def id2unit(self, id):
        return super().id2unit(id)

    def unit2id(self, unit):
        return super().unit2id(unit)

    def build_vocab(self, dataset_path):
        pos = []
        files = glob(dataset_path + '/*.conllu_bio')
        for file in files:
            fr = open(file, 'r', encoding='utf')
            for line in fr.readlines():
                line = line.strip()
                if line == '':
                    continue
                line = line.strip().split('\t')
                pos.append(line[self.idx])
        counter = Counter(pos)
        self._id2unit = VOCAB_PREFIX + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        if self.idx ==5:
            bios=['<PAD>', 'O',
'B-处所',  'I-处所',
'B-工具',  'I-工具',
'B-抽象','I-抽象',
 'B-植物', 'I-植物',
 'B-数目','I-数目',
  'B-时间', 'I-时间',
  'B-建筑', 'I-建筑',
'B-事件','I-事件',
'B-作品', 'I-作品',
'B-动物', 'I-动物',
'B-药物','I-药物',
'B-心理','I-心理',
'B-材料', 'I-材料',
'B-生理', 'I-生理',
'B-食物', 'I-食物',
'B-属性',  'I-属性',
'B-机构','I-机构',
'B-人', 'I-人',
'B-交通工具', 'I-交通工具',
'B-搭配','I-搭配',
'B-货币','I-货币',
'B-生理','I-生理',
'B-信息','I-信息',
'B-衣物','I-衣物',
'<start>', '<eos>']
            self._id2unit = bios
        self._unit2id = {w: i for i, w in enumerate(self._id2unit)}

class KnowVocab(BaseVocab):
    def __init__(self, data=None,idx=3, cutoff=0,vocab_path=None, pos_vocab=None):
        super().__init__(data, idx=idx,cutoff=cutoff,vocab_path=vocab_path,pos_vocab=pos_vocab)

    def id2unit(self, id):
        return super().id2unit(id)

    def unit2id(self, unit):
        return super().unit2id(unit)

    def build_vocab(self, dataset_path):
        if self.vocab_path.exists(): # 为了方便邻接图
            self._id2unit=torch.load(self.vocab_path)
        else:
            # know_label = ['<PAD>', '<UNK>', '<ROOT>', '_', '搭配', '植物', '抽象', '事件', '心理', '处所', '作品',
            #               '生理', '工具', '药物', '货币', '交通工具', '建筑', '机构', '衣服', '数目', '动物',
            #               '材料', '时间', '信息', '食物', '人', '衣物', '属性']
            know_label=['<PAD>', '<ROOT>','<UNK>', 'O',
'B-处所',  'I-处所',
'B-工具',  'I-工具',
'B-抽象','I-抽象',
'B-植物', 'I-植物',
'B-建筑', 'I-建筑',
'B-作品', 'I-作品',
'B-动物', 'I-动物',
'B-药物','I-药物',
'B-心理','I-心理',
'B-材料', 'I-材料',
'B-生理', 'I-生理',
'B-食物', 'I-食物',
'B-属性',  'I-属性',
'B-机构','I-机构',
'B-人', 'I-人',
'B-交通工具', 'I-交通工具',
'B-生理','I-生理',
'B-衣物','I-衣物']
            self._id2unit = know_label
            torch.save(self._id2unit, self.vocab_path)
        self._unit2id = {w: i for i, w in enumerate(self._id2unit)}


class RelVocab(BaseVocab):
    def __init__(self, data=None,idx=8, cutoff=0,vocab_path=None):
        super().__init__(data, idx=idx,cutoff=cutoff, vocab_path=vocab_path)

    def id2unit(self, id):
        return super().id2unit(id)

    def unit2id(self, unit):
        return super().unit2id(unit)

    def build_vocab(self, dataset_path):
        if self.vocab_path.exists(): # 为了能够复现结果，必须这样
            self._id2unit=torch.load(self.vocab_path)
        else:
            rel = []
            files = glob(dataset_path + '/*.conllu')
            for file in files:
                fr = open(file, 'r', encoding='utf')
                for line in fr.readlines():
                    line = line.strip()
                    if line == '':
                        continue
                    line = line.strip().split('\t')
                    arcs = line[self.idx].split('|')
                    for arc in arcs:
                        rel.append(arc.split(':')[1])
            counter = Counter(rel)
            self._id2unit = VOCAB_PREFIX + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
            torch.save(self._id2unit,self.vocab_path)
        self._unit2id = {w: i for i, w in enumerate(self._id2unit)}


    def parse_to_sent_batch(self, inputs):
        sents = []
        for s in inputs:
            words = []
            for w in s:
                arc = []
                for a in w:
                    head = str(a[0])
                    deprel = self.id2unit(a[1])
                    arc.append([head, deprel])
                if len(arc) == 1:
                    string = ':'.join(arc[0])
                    words.append(string)
                else:
                    string = ''
                    for item in arc:
                        string += ':'.join(item) + '|'
                    words.append(string[:-1])
            sents.append(words)
        return sents


if __name__ == '__main__':
    vocab = RelVocab('../dataset')
    # print(vocab._id2unit)
    print(len(vocab))
    print(vocab.map(['Cons', 'Accd', 'dMann', 'TDur', 'Lini', 'Freq', 'eConc', 'rProd']))
    print(vocab.unmap([66, 67, 68, 69, 70, 71, 72, 73]))
    print(vocab.unit2id('dMann'))


    # import pathlib

    # conllu_file_path='..\dataset\sdp_mix_train.conllu'
    # abs_path, file_name = pathlib.Path(conllu_file_path).cwd(), pathlib.Path(conllu_file_path).name  # cwd获取当前路径
    # print(abs_path)
    # print(file_name)
    # cashed_name="cached_"+file_name.strip('sdp_')
    # print(type(abs_path))
    # cached_examples_file = abs_path /cashed_name
    # print(cached_examples_file)
    # if cached_examples_file.exists():
    #     print('2222')
    # f=abs_path/file_name
    # print(f)
    # if f.exists():
    #     print('1111')
