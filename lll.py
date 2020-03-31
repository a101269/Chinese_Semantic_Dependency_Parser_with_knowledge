import re
import random
fr = open("G:\corpus\SDG\medical\medical3300.conll", 'r',encoding="utf")
fw = open('G:\corpus\SDG\medical\medical_replace3300.conll', mode='w', encoding='utf')
'''
m 比例
nt 日期
q 单位
3	所	所	u	u	_	4	mRang	_	_
'''
pattern = re.compile(r'(\[#比例#)')
for line in fr.readlines():
    line = line.strip()
    matchObj = re.search(pattern, line)   # 31	[#比例#	[#比例#	[#比例#	[#比例#	_	29	Nini	_	_
    if matchObj:
        print(matchObj[0])
        if matchObj[0]=='[#比例#':
            line=line.split('\t')
            line[1]= str(random.randint(0,50))+'%'
            line[3]='m'
            line[2] = line[1]
            line[4]=line[3]
        elif matchObj[0]=='[#日期#':
            line=line.split('\t')
            line[1]= str(random.uniform(1988,2020))+'年'
            line[3]='nt'
            line[2] = line[1]
            line[4]=line[3]
        elif matchObj[0]=='[#单位#':
            line=line.split('\t')
            line[1]= 'mg/l'
            line[3]='q'
            line[2] = line[1]
            line[4]=line[3]
        line='\t'.join(line)+'\n'
        fw.write(line+'\n')
    else:
        fw.write(line + '\n')

from utils.utils import init_logger,logger

init_logger(log_file='yuqing.log')
logger.info('   觉得你人好好看，而且声音很好听，人也很热情，能和你组cp太开心   ')