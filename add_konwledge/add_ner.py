#    Author:  a101269
#    Date  :  2020/3/24

import os
from glob import glob
from pyltp import Postagger, Segmentor, NamedEntityRecognizer

LTP_DATA_DIR = 'D:\\ltp_data'  # ltp模型目录的路径
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')

segmentor = Segmentor()
postagger = Postagger()  # 初始化实例
recognizer = NamedEntityRecognizer()  # 初始化实例
postagger.load(pos_model_path)  # 加载模型
segmentor.load(cws_model_path)
recognizer.load(ner_model_path)  # 加载模型

def conll(fr):
    name=fr.split('\\')[-1].split('.')[0]
    print(name)
    fr = open(fr, 'r', encoding="utf")
    fw = open('.\\data\\'+name+'.conllu_ner', mode='w', encoding='utf')
    sents=fr.read().split('\n\n')

    for sent in sents:
        words=[]
        postags=[]
        lines = sent.split('\n')
        for line in lines:
            if not line:
                continue
            line=line.split('\t')
            words.append(line[1])
            # postags.append(line[3])
        # words = segmentor.segment(line)  # 分词
        postags = postagger.postag(words)  # 词性标注""
        netags = recognizer.recognize(words, postags)  # 命名实体识别
        for index, pos in enumerate(postags):
            fw.write(str(index + 1) + "\t" + words[index] + "\t" + words[
                index] + "\t" + pos + "\t" + pos + "\t_\t_\t_\t_\t_\n")  # 写完通过\n进行换行
        fw.write(' '.join(netags)+"\n")
    fw.close()


if __name__ == '__main__':

    files = glob('D:\projects\Parser_with_knowledge\dataset\*.conllu')
    for file in files:
        print(file)
        conll(file)

    postagger.release()  # 释放模型
    segmentor.release()
    recognizer.release()  # 释放模型