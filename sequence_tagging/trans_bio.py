#    Author:  a101269
#    Date  :  2020/4/23
from glob import glob

def trans_to_bio(fr):
    inject_entity_num=0
    name=fr.split('\\')[-1].split('.')[0]
    print(name)
    fr = open(fr, 'r', encoding="utf")
    fw = open('.\\'+name+'.conllu_bio', mode='w', encoding='utf')
    sents=fr.read().split('\n\n')
    sent_num=len(sents)
    for sent_id,sent in enumerate(sents):
        words=[]
        postags=[]
        entities=[]
        head0=[]
        rel0=[]
        rels=[]
        pre_label=None

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
            entities.append(line[9])
        sent_bio=[]
        for index, entity in enumerate(entities):
            if entity !=pre_label and index!=len(entities)-1:
                pre_label=entity
                if entity!='_' and entity!='搭配':
                        sent_bio.append( 'B-' +entity)
                else:
                        sent_bio.append('O')
            elif entity ==pre_label:
                if entity!='_' and entity!='搭配':
                        sent_bio.append( 'I-' +entity)
                else:
                        sent_bio.append('O')
            elif index==len(entities)-1:
                if entity!='_' and entity!='搭配':
                        sent_bio.append( 'I-' +entity)
                else:
                        sent_bio.append('O')
        # print(sent_bio)
        for index, entity in enumerate(entities):
            write_line=str(index + 1) + "\t" + words[index] + "\t" + words[index] + "\t" + postags[index] + "\t" + postags[index] + "\t"+ sent_bio[index]+"\t"+ head0[index]+"\t"+rel0[index]+"\t"+ rels[index] +'\t'+entities[index]

            fw.write(write_line+"\n")
        fw.write('\n')
    fw.close()
    print("加入知识的词数 ："+str(inject_entity_num))


if __name__ == '__main__':
    files = glob('G:\研究生\parser文献\EMNLP2020\conllu_ner\*.conllu_ner')
    for file in files:
        print(file)
        trans_to_bio(file)

