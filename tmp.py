# import re
#
# pattern1 = re.compile(r'(^([零一二三四五六七八九十百千万]+)(分之|点)[零一二三四五六七八九十百千万]*(点?)([零一二三四五六七八九十百千万]+$))')
# pattern2 = re.compile('(((([一二三四五六七八九十]+)|(\d{2,4}))[-/年])?((([一二三四五六七八九十]+)|(\d{1,2}))[-/月])((([一二三四五六七八九十]+)|(\d{1,2}))[日号]*)?)')
# pattern3= re.compile('((([一二三四五六七八九十]+)|(\d{1,2}))[日号]+)|((([一二三四五六七八九十]+)|(\d{2,4}))年)')
# str='1999-3的是有不同哦'
#
# obj= re.search(pattern2, str)
# if obj:
#     print(obj)
#
import torch
pad_zero = torch.LongTensor(4,5)
pad_zero*=0
tag_seq=torch.tensor([[6,6, 10, 1, 1],[5, 5, 0, 2,10],[3, 3, 4, 4, 1],[3, 0, 0, 4, 1]])
print(tag_seq)
tag_seq = torch.where(tag_seq >= 4, tag_seq, pad_zero)
print(tag_seq)