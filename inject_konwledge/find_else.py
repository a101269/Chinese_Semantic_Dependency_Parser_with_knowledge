#    Author:  a101269
#    Date  :  2020/5/8



fw=open('else.txt','w')

pass_own=['音乐作品','网络小说','娱乐作品','言情小说','娱乐','流行音乐','小说作品','娱乐人物','歌词','出版物','书籍', '文学作品','音乐','电影','文学','软件',
          '美术作品','艺术作品','美术','互联网','网站','游戏','娱乐','电视剧','科学','字词,成语,语言','词语','字词,语言']
line_num=0
with open('kg_9minllion', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            subj,obje = line.strip().split("\t")
        except:
            pass
        value = obje
        for p in pass_own:
            if p in value:
                fw.write(subj+'\t其他\n')
                line_num+=1
                if line_num>1000:
                    break

