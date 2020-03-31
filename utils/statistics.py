#    Author:  a101269
#    Date  :  2020/3/20

from glob import glob
def number_sents(fr):
    fr = open(fr, 'r', encoding="utf")
    lines = fr.read()
    sents = lines.strip().split("\n\n")
    sent_num=len(sents)
    great_num=0
    max_len=0
    for sent in sents:
        ori_sent=[]
        sent=sent.split('\n')
        for word in sent:
            chars=word.split('\t')[1]
            ori_sent.append(chars)
        ori_sent=''.join(ori_sent)
        if len(ori_sent)>246:
            great_num+=1
        if len(ori_sent)>max_len:
            max_len=len((ori_sent))
    print(max_len)
    fr.close()
    return sent_num,great_num

if __name__ == '__main__':
    files = glob('F:\parser_project\Parser_with_knowledge\dataset\*.conllu')
    for file in files:
        print(file)
        sent_num, great_num=number_sents(file)
        print(sent_num,great_num)

# grep -r "Model config" /data/private/ldq/anaconda3/lib/python3.6/site-packages/transformers