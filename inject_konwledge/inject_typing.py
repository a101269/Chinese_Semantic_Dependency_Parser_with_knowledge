#    Author:  a101269
#    Date  :  2020/3/30

from tqdm import tqdm
from glob import glob
import re
import jieba
own_pos={'处所':'N','机构':'N','人':'N','动物':'N','植物':'N','工具':'N','交通工具':'N','药物':'N','理论':'N','作品':'N','食物':'N','材料':'N','建筑':'N',
         '抽象':'N','衣物':'N','心理':'N','生理':'N',}
pos_map={'NN':'N','NR':'N','NT':'N'}

own_kvs={'处所':'处所','城市':'处所','地点':'处所','地名':'处所','地址':'处所','地理':'处所','空间':'处所','国籍':'处所','位置':'处所','地方':'处所',
          '天体':'处所','地表':'处所','地貌':'处所','行政区划':'处所','陵墓':'处所','山脉':'处所','河流':'处所','陆地':'处所','水域':'处所','大地':'处所',
          '日期':'时间', '朝代':'时间',
          '数目':'数目','数量': '数目','比例':'数目',
          '机构': '机构','组织': '机构', '公司': '机构', '团体': '机构','集体':'机构','医院':'机构',
          '政府':'机构','军队':'机构','工厂':'机构','国家': '机构',
          '人物':'人', '人名':'人','亲属':'人','职业':'人', '职务':'人','身份':'人','人类':'人','人':'人',
          '动物': '动物','兽': '动物', '昆虫':'动物', '鱼':'动物','爬行动物':'动物','微生物':'动物','牲畜':'动物',
          '植物':'植物', '庄稼':'植物', '树':'植物', '草':'植物','花':'植物',
          '工具':'工具','用具':'工具','器械':'工具','器材':'工具','乐器':'工具', '电器':'工具',
          '器具':'工具','炊具':'工具','厨房用具':'工具','运动器械':'工具','文具':'工具',
          '家具':'工具','化妆品':'工具','硬件':'工具','机器':'工具',
          '交通工具':'交通工具','车辆':'交通工具','船':'交通工具','飞机':'交通工具','飞行器':'交通工具','车船':'交通工具',
          '属性': '属性', '颜色': '属性', '外观': '属性', '外形': '属性', '外貌': '属性', '性格': '属性',
          '材料':'材料','矿物': '材料', '化合物': '材料',
          '药物':'药物', '药品':'药物',
          '作品':'作品','软件':'作品', '符号':'作品','证书':'作品','票据':'作品',
          '理论':'抽象','法规':'抽象','知识':'抽象','律法':'抽象','信息':'信息',
          '生理': '生理', '疾病':'生理',
          '心理':'心理','情感':'心理', '意识':'心理', '人性':'心理','动机':'心理',
          '事件': '事件', '事务': '事件', '事情':'事件','手术':'事件', '过程':'事件','活动':'事件',
          '时间': '时间','学校':'机构','货币':'货币','书':'作品',
          '衣服':'衣服', '衣物': '衣物', '服饰': '衣物',
          '食物': '食物','食品': '食物','水果':'食物','建筑':'建筑',
          '音乐':'作品', '电影':'作品','创作物':'作品','出版物':'作品'# '性能': '性能','方位':'方位',
         #'搭配': '搭配'搭配带序号的，不能重复,# '生物': '生物',
         #'气象': '现象', '自然现象': '现象',
          }
# key_word=own_kvs.keys()
key_word=[
          '国家','国籍','处所', '城市', '地点', '地名', '地址', '地理', '空间', '天体', '地表', '地貌','河流', '陆地','水域','日期','山脉', '陵墓','行政区划',
          '位置','大地','地方','朝代',
          '数目', '数量', '比例', '机构', '组织', '公司', '团体', '集体', '医院', '政府', '军队', '工厂',
          '动物','牲畜', '兽', '昆虫', '鱼', '爬行动物', '微生物', '植物', '庄稼', '树', '草', '花', '水果',
          '交通工具','车船', '车辆', '船', '飞机', '飞行器',
          '工具', '用具', '器械', '器材', '乐器', '电器', '器具', '炊具', '厨房用具', '运动器械', '文具', '家具', '电器',
          '化妆品', '硬件', '机器', '属性', '颜色', '外观', '外形',
          '外貌', '性格', '材料', '矿物', '化合物', '药物','药品', '药','作品', '软件', '符号', '证书', '票据', '理论', '法规','知识', '信息',
          '生理', '疾病', '心理', '情感', '意识', '人性', '动机', '事件', '事务', '事情', '手术', '过程', '活动', '事情', '食物','食品', '建筑',
          '人','人物',  '人名', '亲属', '职业', '职务', '身份', '人类','衣服', '衣物', '服饰','时间','学校','货币','书','音乐' '电影','创作物','出版物',
]
over_write_owns=['时间','事件','数目','信息','货币','搭配']
pass_own=['音乐作品','网络小说','娱乐作品','言情小说','娱乐','流行音乐','小说作品','娱乐人物','歌词','出版物','书籍', '文学作品','音乐','电影','文学','软件',
          '美术作品','艺术作品','美术','互联网','网站','游戏','娱乐','电视剧','科学','字词,成语,语言','词语','字词,语言']

pattern1 = re.compile(r'(^([零一二三四五六七八九十百千万]+)(分之|点)[零一二三四五六七八九十百千万]*(点?)([零一二三四五六七八九十百千万]+$))')
pattern2 = re.compile('(((([一二三四五六七八九十]+)|(\d{2,4}))[-/年])?((([一二三四五六七八九十]+)|(\d{1,2}))[-/月])((([一二三四五六七八九十]+)|(\d{1,2}))[日号]*)?)')
pattern3= re.compile('((([一二三四五六七八九十]+)|(\d{1,2}))[日号]+)|((([一二三四五六七八九十]+)|(\d{2,4}))年)')

def create_lookup_table():
    lookup_table = {}
    with open('../corpus/kg_entity_typing', 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            try:
                subj, obje = line.strip().split("\t")
            except:
                print("[KnowledgeGraph] Bad spo:", line)
            value = []
            pass_flag=False
            for ib_idx,ob in enumerate(obje.split(',')):
                if ob in pass_own:
                    pass_flag=True
                    continue
                if pass_flag and ob not in key_word:
                    continue
                    # value=None
                    # break
                value.append(ob)
            if value==None:
                continue
            value=','.join(value)
            # if len(value)>16:
            #     value=value[:16]
            if value and len(subj)>=2:
                if subj in lookup_table.keys():
                    lookup_table[subj].append(value)
                else:
                    lookup_table[subj] = [value]
    return lookup_table

lookup_table = create_lookup_table()

user_dict='/data/private/ldq/projects/data_ner/mydict'
jieba.load_userdict(user_dict)

# tokenizer = pkuseg.pkuseg(model_name="default", postag=False, user_dict=segment_vocab)
# fw_pre_word_list= open('./entity_typeing_words', mode='w', encoding='utf')
def conll(fr):
    inject_entity_num=0
    name=fr.split('/')[-1].split('.')[0]
    print(name)
    is_text = False
    if 'text' in name:
        is_text=True
    fr = open(fr, 'r', encoding="utf")
    fw = open('./'+name+'.conllu_ner', mode='w', encoding='utf')
    sents=fr.read().split('\n\n')
    sent_num=len(sents)
    for sent_id,sent in enumerate(tqdm(sents)):
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
        split_sent= jieba.lcut(ori_sent)
        for i,token in enumerate(split_sent):
            if i==0:
                boundary_new.append(len(token))
            else:
                boundary_new.append(boundary_new[-1]+len(token))
            entitie = lookup_table.get(token) # 查询实体对应标签，entity_typing在此进行
            if entitie==None:
                continue
            # fw_pre_word_list.write(token+'\n')
            entitie = ','.join(list(entitie))
            # if token=='火车':
            #     print(entitie)
            entities.append((token,entitie))
            entity_s_e.append((boundary_new[-1]-len(token)+1,boundary_new[-1])) # 从1开始
        entity_num=0
        for index, pos in enumerate(postags):
            entity_info = "_"
            one_word=False # 是否是单个词的实体
            flag = False # 是否能找到本体本体
            for seidx, (s,e) in enumerate(entity_s_e):
                # if words[index]=='澳大利亚':
                #     print(s,e)
                #     print(boundary_ori[index])
                if index==0 and e==boundary_ori[index]:
                    for kw in key_word:
                        if kw in entities[seidx][1]:
                            entity_info= kw
                            flag = True
                            one_word=True
                            break
                    if not flag:
                        continue
                    entity_s_e.pop(seidx)
                    entities.pop(seidx)
                    entity_num += 1
                elif index>0 and s==boundary_ori[index-1]+1 and e==boundary_ori[index]:
                    for kw in key_word:
                        if kw in entities[seidx][1]:
                            entity_info= kw
                            flag = True
                            one_word = True
                            break
                    if not flag:
                        continue
                    entity_s_e.pop(seidx)
                    entities.pop(seidx)
                    entity_num += 1
                elif s<=boundary_ori[index] and e>boundary_ori[index]:
                    # if not is_text:
                    for kw in key_word:
                        if kw in entities[seidx][1]:
                            entity_info= kw
                            flag = True
                            break
                    if not flag:
                        entity_info = '搭配'#+str(entity_num)
                elif s<boundary_ori[index] and e==boundary_ori[index]:
                    # if not is_text:
                    for kw in key_word:
                        if kw in entities[seidx][1]:
                            entity_info= kw
                            flag = True
                            break
                    if not flag:
                        entity_info = '搭配'#+str(entity_num)
                    entity_s_e.pop(seidx)
                    entities.pop(seidx)
                    entity_num += 1
            if entity_info != '搭配':
                inject_entity_num+=1
                if entity_info!='_':
                    entity_info=own_kvs.get(entity_info)
                if one_word and entity_info in own_pos and pos_map.get(pos) !=own_pos[entity_info]:
                    entity_info='_'
                if one_word and entity_info not in own_pos :
                    entity_info='_'
                if not one_word and entity_info =='属性' :
                    entity_info='_'
                if entity_info =='' or  entity_info ==None or entity_info in over_write_owns:
                    entity_info='_'
                if re.search(pattern1, words[index]):
                    entity_info='_'
                elif re.search(pattern2, words[index]) or re.search(pattern3, words[index]):
                    entity_info='_'
            # print(entity_info+ words[index])
            write_line=str(index + 1) + "\t" + words[index] + "\t" + words[index] + "\t" + pos + "\t" + pos + '\t_\t'+ head0[index]+"\t"+rel0[index]+"\t"+ rels[index] +"\t"+entity_info
            # if write_line.split('\t')!=10:
            #     print(write_line)
            #     break
            fw.write(write_line+"\n")
        fw.write('\n')
    fw.close()
    print("加入知识的词数 ："+str(inject_entity_num))


if __name__ == '__main__':
    files = glob('/data/private/ldq/projects/Parser_with_knowledge/dataset/*.conllu')
    for file in files:
        print(file)
        conll(file)

