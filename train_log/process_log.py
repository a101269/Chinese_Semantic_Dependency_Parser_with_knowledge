#    Author:  a101269
#    Date  :  2020/5/14
#
# [2020-05-14 10:08:33,999 INFO] LAS ,UAS in epoch1,step681:0.0000,0.0089
# [2020-05-14 10:13:04,164 WARNING] epoch1, step:1362-----LAS:0.0590,UAS:0.2436,loss 7.3370
fr=open('lstm3tran2','r')
result={0:0}
last_epoch=0
half=False
for line in fr.readlines():

    if 'WARNING' in line:
        epoch=int(line.split('epoch')[1].split(', step:')[0])
        if epoch!=last_epoch+1 :
            if epoch == last_epoch :
                if not half:
                    LAS = line.split('LAS:')[1].split(',UAS')[0]
                    result[epoch+0.5] = float(LAS)
                    half=True
            # else:
            #     half=False
            #     result[last_epoch+1] = result[last_epoch]
        else:
            half=False
            LAS=line.split('LAS:')[1].split(',UAS')[0]
            result[epoch]=float(LAS)

        last_epoch=epoch
    else:
        epoch=int(line.split('epoch')[1].split(',step')[0])
        if epoch!=last_epoch+1 :
            if epoch == last_epoch :
                if not half:
                    LAS = line.split('step')[1].split(':')[1].split(',')[0]
                    result[epoch+0.5] = float(LAS)
                    half=True
            # else:
            #     half=False
            #     result[last_epoch+1] = result[last_epoch]
        else:
            half=False
            LAS = line.split('step')[1].split(':')[1].split(',')[0]
            result[epoch]=float(LAS)

        last_epoch=epoch

print(result)
x=[]
y=[]
for k,v in result.items():
    x.append(k)
    y.append(v)
print(x,y)

