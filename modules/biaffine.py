#    Author:  a101269
#    Date  :  2020/3/10

import torch
import torch.nn as nn
import torch.nn.functional as F

class PairwiseBilinear(nn.Module):
    ''' A bilinear module that deals with broadcasting for efficient memory usage.
    Input: tensors of sizes (N x L1 x D1) and (N x L2 x D2)
    Output: tensor of size (N x L1 x L2 x O)'''
    def __init__(self, input1_size, input2_size, output_size, bias=True):  #  self.W_bilin = PairwiseBilinear(input1_size + 1, input2_size + 1, output_size)
        super().__init__()   # 7        7          7

        self.input1_size = input1_size
        self.input2_size = input2_size
        self.output_size = output_size

        self.weight = nn.Parameter(torch.Tensor(input1_size, input2_size, output_size))
        self.bias = nn.Parameter(torch.Tensor(output_size)) if bias else 0

    def forward(self, input1, input2): # # [2*3*7]，[2*3*7]
        input1_size = list(input1.size())
        input2_size = list(input2.size())
        output_size = [input1_size[0], input1_size[1], input2_size[1], self.output_size]
        # print("self.W_bilin(input1, input2      self.W_bilin(input1, input2    self.W_bilin(input1, input2")
        # print(input1.view(-1, input1_size[-1]).shape)
        # print(self.weight.view(-1, self.input2_size * self.output_size).shape)
        # ((N x L1) x D1) * (D1 x (D2 x O)) -> (N x L1) x (D2 x O)
                                 #in1    6*7   3100*401                      # in1 * (in2 * out) 维   7*49   401*56140
        intermediate = torch.mm(input1.view(-1, input1_size[-1]), self.weight.view(-1, self.input2_size * self.output_size))  # mm 矩阵乘法
        # (N x L2 x D2) -> (N x D2 x L2)
        input2 = input2.transpose(1, 2)  # 转置 2*7*3
        # (N x (L1 x O) x D2) * (N x D2 x L2) -> (N x (L1 x O) x L2)
        output = intermediate.view(input1_size[0], input1_size[1] * self.output_size, input2_size[2]).bmm(input2)  # bmm 矩阵乘法
        # (N x (L1 x O) x L2) -> (N x L1 x L2 x O)
        output = output.view(input1_size[0], input1_size[1], self.output_size, input2_size[1]).transpose(2, 3)
        # torch.Size([73, 50, 50, 1]) 是否是头节点;torch.Size([73, 50, 50, 140]) 140个标签
        return output

class BiaffineScorer(nn.Module):
    def __init__(self, input1_size, input2_size, output_size):
        super().__init__()
        self.W_bilin = nn.Bilinear(input1_size + 1, input2_size + 1, output_size)  # Applies a bilinear transformation to the incoming data: y = x1 * W * x2 + b`
                                                    # +1原因：输入数据调用此函数前最后维度上均与1进行了拼接，维度增加了1
        self.W_bilin.weight.data.zero_()
        self.W_bilin.bias.data.zero_()

    def forward(self, input1, input2):
        # print(input1)
        # print(input1.size())
        input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)], len(input1.size())-1) # cat第一个参数类型元组或列表均可 any python sequence of tensors
        input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)], len(input2.size())-1)  # 为什么拼接？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
        return self.W_bilin(input1, input2)                                    # 拼接结果最后一维 即hidden维度所在层次增加了一位 1

class PairwiseBiaffineScorer(nn.Module):  # 6   6   7
    def __init__(self, input1_size, input2_size, output_size):
        super().__init__()                   # 7             7                7
        self.W_bilin = PairwiseBilinear(input1_size + 1, input2_size + 1, output_size)  # 这里的size为输入数据的最后一维 hidden_size

        self.W_bilin.weight.data.zero_()
        self.W_bilin.bias.data.zero_()
     #  unlabeled(lstm_outputs，lstm_outputs)
    def forward(self, input1, input2):# [2*3*6],[2*3*6]           [:-1]切片索引到-1，则不含-1，因此少了最后一维
        input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)], len(input1.size())-1)  # [ 2*3*7],[2*3*6]与[2*3*1] 在最后一维拼接，
        input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)], len(input2.size())-1)  # [2*3*7]

        # a=self.W_bilin(input1, input2)
        # print(a.shape)   # torch.Size([73, 50, 50, 1]) torch.Size([73, 50, 50, 140]) 是否是头节点，140个标签
        return self.W_bilin(input1, input2)

class DeepBiaffineScorer(nn.Module):  #  self.deprel = DeepBiaffineScorer(2 * self.args['hidden_dim'], 2 * self.args['hidden_dim'], self.args['deep_biaff_hidden_dim'], len(vocab['graph']), pairwise=True, dropout=args['dropout'])
                    #    4          5             6            7
    def __init__(self, input1_size, input2_size, hidden_size, output_size, hidden_func=F.relu, dropout=0, pairwise=True):
        super().__init__()
        self.W1 = nn.Linear(input1_size, hidden_size)
        self.W2 = nn.Linear(input2_size, hidden_size)
        self.hidden_func = hidden_func
        if pairwise:                                # 6           6          7
            self.scorer = PairwiseBiaffineScorer(hidden_size, hidden_size, output_size)
        else:
            self.scorer = BiaffineScorer(hidden_size, hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input1, input2):   # dropout,  tanh, linear  结果 计算得分
        # print(input1)           #   2*3*6                                              2*3*6
        return self.scorer(self.dropout(self.hidden_func(self.W1(input1))), self.dropout(self.hidden_func(self.W2(input2))))

if __name__ == "__main__":
    x1 = torch.randn(2, 3, 4)
    x2 = torch.randn(2, 3, 5)
    scorer = DeepBiaffineScorer(4, 5, 6, 7)
    # print(scorer(x1, x2))
    res = scorer(x1, x2)
    print(res)
    # print(res.size())


