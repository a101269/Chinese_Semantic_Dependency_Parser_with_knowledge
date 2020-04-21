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
    # print(max_len)
    fr.close()
    return sent_num,great_num
'''
 news.train.conll:8301句
 news.valid.conll:534句
 news.test.conll:1233句

 text.train.conll:10754句
 text.valid.conll:1535句
 text.test.conll:3073句


 
'''

if __name__ == '__main__':
    files = glob('G:\corpus\SDG\conllu_file\\new_ccl_coarse_dataset\*.conllu')
    sum_num=0
    for file in files:
        sent_num, great_num=number_sents(file)
        name=file.split('\\')[-1]
        print(name+':'+str(sent_num)+'句')
        sum_num+=sent_num
    print('总句数'+str(sum_num))

