#    Author:  a101269
#    Date  :  2020/5/6

from tqdm import tqdm
from glob import glob
import re

def add_label(fr,vocabs):
    fr=open(fr,'r',encoding='utf')
    for line in fr.readlines():
        word=line.strip('\n').strip()
        vocabs.add(word)
    return vocabs

# location，organization，anim,plant,food ,drug,bill，book，building,clothes,event
if __name__ == '__main__':

    # transform_and_save('entity_typing\source_data\\药品名称大全.scel','entity_typing\\target_data\\drug\药品.txt')
    # 心理psychology
    #
    # fr = open('G:\corpus\KG_data\w2o_info.txt', 'r', encoding='utf')
    # fw = open('entity_typing\\target_data\\tool\武器.txt', 'w', encoding='utf')
    # for i,line in enumerate(fr.readlines()):
    #     word = line.strip('\n').split('\t')
    #     if word[-1]== '武器':#in word[-1]:#word[-1]=='关系':#or word[-1]=='残疾':
    #         fw.write(word[0]+'\n')

    files = glob('target_data/drug/*.txt')
    vocabs=set()
    for file in files:
        print(file)
        vocabs=add_label(file,vocabs)
        print(len(vocabs))
    fw=open('dataset/药品.txt','w',encoding='utf')
    ending=['局','会','院','学','厂' ,'团','菌','器','委','毒','市','机','备','司','厂','团','业']    # 食品饮料存在 设备 省市 机  # 材料存在 病毒 菌 # 矿产材料 设备 机
    # 工具类 代理 加工 合合作 废料 材料 煤 油 沥 青
    # 建筑物 系统 设备 频道  器 炉 机 堆
    for word in vocabs:
        if word=='':
            continue
        # fw.write(word + '\t' + '药品' + '\n')
        if word[-1] in ending:#  or (word[-1]=='厅' and word[-1]!='餐') or (word[-1]=='所' and word[-1]!='会'):
            continue
        else:
            fw.write(word + '\t' + '药品' + '\n')

