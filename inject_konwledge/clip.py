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
            if len(token)<2:
                continue
            entitie = lookup_table.get(token)
            if entitie==None:
                continue
            entitie = ','.join(list(entitie))
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
                if entity_info =='' or  entity_info ==None:
                    entity_info='_'
                if re.search(pattern1, words[index]):
                    entity_info='数目'
                elif re.search(pattern2, words[index]) or re.search(pattern3, words[index]):
                    entity_info='时间'
            # print(entity_info+ words[index])
            write_line=str(index + 1) + "\t" + words[index] + "\t" + words[index] + "\t" + pos + "\t" + pos + '\t_\t'+ head0[index]+"\t"+rel0[index]+"\t"+ rels[index] +"\t"+entity_info
            # if write_line.split('\t')!=10:
            #     print(write_line)
            #     break
            fw.write(write_line+"\n")
        fw.write('\n')
    fw.close()
    print("加入知识的词数 ："+str(inject_entity_num))

files = glob('/data/private/ldq/projects/Parser_with_knowledge/dataset/*.conllu')
for file in files:
    print(file)
    conll(file)
