#    Author:  a101269
#    Date  :  2020/5/26


def mark_error(event_lab):
    err_pos={}
    err_rel={}
    gold_eve={}
    err_sum=0
    gold_sum=0
    right_sum=0
    for k in event_lab:
        err_rel[k]=0
        gold_eve[k]=0
    fr = open('D:\研究生\Parser_with_knowledge\dataset\\result\\news_ner_eval_temp.conllu', 'r', encoding="utf")
    gold=open('D:\研究生\Parser_with_knowledge\dataset\sdp_news_test.conllu', 'r', encoding="utf")
    fw=open('news_bad_case','w',encoding='utf')
    lines = fr.read()
    golds=gold.read()
    sents = lines.strip().split("\n\n")
    gold_sents=golds.strip().split("\n\n")
    for i,sent in enumerate(sents):
        sent = sent.split('\n')
        gold_sent=gold_sents[i].split('\n')
        for j,word in enumerate(sent):
            items=word.split('\t')
            rels = items[-2]
            gold_rels=gold_sent[j].split('\t')[-2]
            if rels!=gold_rels:
                items[-2]=rels+'---!!!!!---'+gold_rels
                pos=items[3]
                if pos not in err_pos.keys():
                    err_pos[pos]=1
                else:
                    err_pos[pos] += 1
            fw.write('\t'.join(items) + "\n")
            rels=rels.split('|')
            gold_rels=gold_rels.split('|')
            for gold_rel in gold_rels:
                label = gold_rel.split(':')[1]
                if label in event_lab:
                    gold_sum+=1
                    gold_eve[label]+=1
            for rel in rels:
                label = rel.split(':')[1]
                if label in event_lab and rel not in gold_rels:
                    err_rel[label]+=1
                    err_sum+=1
                elif label in event_lab and rel in gold_rels:
                    right_sum+=1

        fw.write('\n')
    fw.close()
    fr.close()
    print(err_pos)
    print(err_rel)
    print(gold_eve)
    print(gold_sum)
    print(err_sum)

    print(right_sum)
    print(err_sum/gold_sum)
    print(right_sum/gold_sum)

if __name__ == '__main__':
    rels='''eCoo
eSelt
eEqu
ePrec
eSucc
eProg
eAdvt
eCau
eResu
eInf
eCond
eSupp
eConc
ePurp
eAban
ePref'''
    event_lab=[]
    for rel in rels.split('\n'):
        event_lab.append(rel)
    print(event_lab)
    mark_error(event_lab)
'''

并列、选择、等同、先行、顺承、递进、转折、原因、结果、推论、条件、假设、让步、目的、割舍、选取、

'''
