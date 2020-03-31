#    Author:  a101269
#    Date  :  2020/3/25
from tqdm import tqdm

lookup_table = {}
owns=set()
with open('ownthink_v2.csv', 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        try:
            subj, pred, obje = line.strip().split(",")
        except:
            pass
            # print("[KnowledgeGraph] Bad spo:", line)
        else:
            value = obje
        if pred=='标签':
            owns.add(value)
            if subj in lookup_table.keys():
                lookup_table[subj].add(value)
            else:
                lookup_table[subj] = set([value])

fw=open('kg_9minllion','w')
for k,v in tqdm(lookup_table.items()):
    fw.write(k+'\t'+','.join(v)+'\n')
fw.close()