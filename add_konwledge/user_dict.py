#    Author:  a101269
#    Date  :  2020/3/30
from tqdm import tqdm
from glob import glob

fw=open('mydict','w',encoding='utf')

def create_lookup_table():
    lookup_table = {}
    with open('../corpus/kg_9minllion', 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            try:
                subj, obje = line.strip().split("\t")
                fw.write(subj + '\n')
            except:
                print("[KnowledgeGraph] Bad spo:", line)
    fw.close()



if __name__ == '__main__':
    create_lookup_table()