#!/usr/bin/env python
# coding: utf-8

# In[3]:


import re
from math import sqrt


# In[10]:


def dataProcess():
    L = []
    U = []
    fi = open("F:\parser_project\Parser_with_knowledge\\train_log\lstm3tran2", "r", encoding='utf-8')
    fi = fi.read()
    lines = fi.strip().split("\n")
    
    for line in lines:
        line = re.sub('\\[.*?\\]','',line).strip()
        if line.startswith('epoch'):
            line = line[line.index('L'):]
            LAS = float(line.split(',')[0].split(':')[1].strip())
            UAS = float(line.split(',')[1].split(':')[1].strip())
        else:
            LAS = float((line.split(':')[1].split(',')[0].strip()))
            UAS = float((line.split(':')[1].split(',')[1].strip()))
        L.append(LAS)
        U.append(UAS)
    print(len(L))
    print(len(U))
    return L,U


# In[11]:


def multipl(a,b):
    sumofab=0.0
    for i in range(len(a)):
        temp=a[i]*b[i]
        sumofab+=temp
    return sumofab


# In[12]:


def corrcoef(x,y):
    n=len(x)
    #求和
    sum1=sum(x)
    sum2=sum(y)
    #求乘积之和
    sumofxy=multipl(x,y)
    #求平方和
    sumofx2 = sum([pow(i,2) for i in x])
    sumofy2 = sum([pow(j,2) for j in y])
    num=sumofxy-(float(sum1)*float(sum2)/n)
    #计算皮尔逊相关系数
    den=sqrt((sumofx2-float(sum1**2)/n)*(sumofy2-float(sum2**2)/n))
    return num/den


# In[13]:


L,U = dataProcess()


# In[15]:

print(L)
print(U)
print(corrcoef(L,U))


# In[ ]:




