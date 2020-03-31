#    Author:  a101269
#    Date  :  2020/3/24
from tqdm import tqdm
import pkuseg
from glob import glob

fr=open('owns','r',encoding='utf')
owns=fr.read().split('\n')
print(len(owns))

key_word=[
'处所', '信息', '自然物', '鸟', '量化属性', '性格', '相对时间', '器具', '电器', '药物', '自然现象', '时段', '身份', '模糊属性', '职业', '爬行动物',
 '票据', '地表', '交通工具', '抽象空间', '具体空间', '事件', '集体', '人类', '非身体构件', '时点', '外形', '专名', '昆虫', '天体', '姓名', '建筑',
  '文具', '机构', '水域物', '人为事物', '化合物', '兽', '事件构件', '时间', '其他抽象事物', '非生物', '计算机软件', '理论', '人名', '地貌', '抽象事物', '化妆品',
   '运动器', '动机', '气象', '创作物', '衣服', '人群', '生物', '身体构件', '方法', '空间', '庄稼', '非生物构件', '衣物', '树', '绝对时间', '可视现象',
    '事性', '用具', '钱财', '矿物', '材料', '过程', '领域', '可食物', '事物', '事理', '属性', '团体', '自然事物', '排泄物', '微生物', '符号', '性能', '物性',
     '计算机硬件', '食物', '作品', '人工物', '植物', '乐器', '可听现象', '技术', '泛称', '服饰', '方位', '具体事物', '构件', '外观', '所处',
 '鱼', '运动器械', '生理', '运动器具', '证书', '法规', '事情', '人性', '亲属', '元素', '个人', '颜色', '地理', '动物', '心理特征', '家具', '情感', '意识',
 '职业','组织机构', '组织','机构',  '地名','地点' '城市', '数量', '疾病', '器械', '称谓', '时间', '职务', '工具', '手术', '车辆', '药物', '用具', '数目', '地点','公司']

pass_own=['音乐作品','网络小说','娱乐作品','言情小说','娱乐','流行音乐','小说作品','娱乐人物','歌词','出版物','书籍', '文学作品','音乐','电影','文学','软件',
          '美术作品','艺术作品','美术','互联网','网站','游戏','娱乐','电视剧','科学','字词,成语,语言','词语','字词,语言']

# 字词,语言  若只是一个词 pass 俩以上加 不在这里进行处理
# 葡萄牙人4600
# save_owns=set()
# for own in owns:
#     for k in key_word:
#         if k in own:
#             save_owns.add(own)


def create_lookup_table():
    lookup_table = {}
    owns=set()
    with open('../corpus/kg_9minllion', 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            try:
                subj, obje = line.strip().split("\t")
            except:
                print("[KnowledgeGraph] Bad spo:", line)
            value = []
            for ob in obje.split(','):
                if ob in pass_own:
                    value=None
                    break
                value.append(ob)
            if value==None:
                continue
            value=','.join(value)
            # if len(value)>16:
            #     value=value[:16]
            if value and len(subj)>1:
                if subj in lookup_table.keys():
                    lookup_table[subj].append(value)
                else:
                    lookup_table[subj] = [value]
    return lookup_table

lookup_table = create_lookup_table()
user_dict= list(lookup_table.keys())
# user_dict='/data/private/ldq/projects/data_ner/mydict'
tokenizer = pkuseg.pkuseg(model_name="default", postag=False, user_dict=user_dict)

def conll(fr):
    name=fr.split('/')[-1].split('.')[0]
    print(name)
    fr = open(fr, 'r', encoding="utf")
    fw = open('./'+name+'.conllu_ner', mode='w', encoding='utf')
    sents=fr.read().split('\n\n')

    for sent in tqdm(sents):
        words=[]
        postags=[]
        entities=[]
        boundary_ori=[]
        boundary_new=[]
        head0=[]
        rel0=[]
        rels=[]
        entity_s_e=[]
        lines = sent.split('\n')
        for i,line in enumerate(lines):
            if not line:
                continue
            line=line.split('\t')
            words.append(line[1])
            postags.append(line[3])
            head0.append(line[6])
            rel0.append(line[7])
            rels.append(line[8])
            if i==0:
                boundary_ori.append(len(line[1]))
            else:
                boundary_ori.append(boundary_ori[-1]+len(line[1]))
        ori_sent=''.join(words)
        split_sent = tokenizer.cut(ori_sent)
        for i,token in enumerate(split_sent):
            if i==0:
                boundary_new.append(len(token))
            else:
                boundary_new.append(boundary_new[-1]+len(token))
            entitie = lookup_table.get(token)

            if entitie==None:
                continue
            entitie = ','.join(list(entitie))
            entities.append((token,entitie))
            entity_s_e.append((boundary_new[-1]-len(token)+1,boundary_new[-1])) # 从1开始

        for index, pos in enumerate(postags):
            entity_info = "_"
            for seidx, (s,e) in enumerate(entity_s_e):
                if index>0 and s==boundary_ori[index-1] and e==boundary_ori[index]:
                    for kw in key_word:
                        if kw in entities[seidx][1]:
                            entity_info= kw
                            entity_s_e.pop(seidx)
                            entities.pop(seidx)
                            break
                        else:
                            entity_info = "_"
                            break
                elif s<=boundary_ori[index] and e>boundary_ori[index]:
                    flag=False
                    for kw in key_word:
                        if kw in entities[seidx][1]:
                            entity_info= kw
                            flag = True
                            break
                    if not flag:
                        entity_info = '词组'+str(seidx)
                    # entity_info=entities[seidx][1]
                elif s<boundary_ori[index] and e==boundary_ori[index]:
                    flag = False
                    for kw in key_word:
                        if kw in entities[seidx][1]:
                            entity_info= kw
                            flag = True
                            break
                    if not flag:
                        entity_info = '词组'+str(seidx)
                    # entity_info=entities[seidx][1]
                    entity_s_e.pop(seidx)
                    entities.pop(seidx)
            fw.write(str(index + 1) + "\t" + words[index] + "\t" + words[index] + "\t" + pos + "\t" + pos + '\t_\t'+ head0[index]+"\t"+rel0[index]+"\t"+ rels[index] +"\t"+entity_info+"\n")
        fw.write('\n')
        # fw.write(' '.join(str(entities))+"\n")
    fw.close()



if __name__ == '__main__':
    owns=set()
    files = glob('/data/private/ldq/projects/Parser_with_knowledge/dataset/*.conllu')
    for file in files:
        print(file)
        conll(file)
