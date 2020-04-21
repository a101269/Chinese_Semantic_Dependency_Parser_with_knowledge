#    Author:  a101269
#    Date  :  2020/3/25
from tqdm import tqdm

lookup_table1 = {}
lookup_table2 = {}
owns=set()

with open('bigcilin_hyper.txt', 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        try:
            subj,obje = line.strip().split(";")
        except:
            pass
        value = obje
        if subj in lookup_table1.keys():
                lookup_table1[subj].add(value)
        else:
                lookup_table1[subj] = set([value])

with open('ownthink_v2.csv', 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        try:
            subj, pred, obje = line.strip().split(",")
        except:
            pass
            # print("[KnowledgeGraph] Bad spo:", line)
        value = obje
        # subj='弥散[汉语词汇]'
        subj=subj.split('[')[0]
        if pred=='标签':
            # owns.add(value)
            if subj in lookup_table2.keys():
                lookup_table2[subj].add(value)
            else:
                lookup_table2[subj] = set([value])

for k1,v1 in lookup_table1.items():
    lookup_table2[k1]=v1

fw=open('kg_9minllion','w')
fw2=open('mydict','w',encoding='utf')
for k,v in tqdm(lookup_table2.items()):
    fw.write(k+'\t'+','.join(v)+'\n')
    fw2.write(k + '\n')
fw.close()
fw2.close()
