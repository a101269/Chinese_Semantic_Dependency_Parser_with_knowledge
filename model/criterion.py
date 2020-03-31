#    Author:  a101269
#    Date  :  2020/3/12
import torch
import torch.nn as nn


def criterion(head_scores, label_scores, head_target, label_target, word_mask, max_len_of_batch, vocabs):
    weights = torch.ones(head_target.size(0), max_len_of_batch, max_len_of_batch, dtype=torch.long,
                         device=head_target.device)
    weights = weights.masked_fill(word_mask.unsqueeze(1), 0)  # 在mask值为1的位置处用value填充
    weights = weights.masked_fill(word_mask.unsqueeze(2), 0)  # torch.int64
    crit_head = nn.BCEWithLogitsLoss(weight=weights.float(), reduction='sum')  # 二分类用的交叉熵，先用Sigmoid给这些值都搞到0-1之间，再计算交叉熵
    # weight：可用于控制各样本的权重，常用作对对齐后的数据进行mask操作（设为0）
    # reduction：控制损失输出模式。设为"sum"表示对样本进行求损失和；设为"mean"表示对样本进行求损失的平均值；而设为"none"表示对样本逐个求损失，输出与输入的shape一样。
    # pos_weight用于设置损失的class权重，用于缓解样本的不均衡问题
    head_loss = crit_head(head_scores, head_target.float())  # 计算损失
    '''
    deprel_target -- type:tensor, shape:(n, m, m)
    deprel_scores -- type:tensor, shape:(n, m, m, c)
    deprel_mask -- tyep:tensor, shape:(n, m)
    '''  # ignore_index=-1 有何用
    crit_rel = nn.CrossEntropyLoss(ignore_index=-1,
                                   reduction='sum')  # nn.CrossEntropyLoss()是nn.logSoftmax()和nn.NLLLoss()的整合,不是我们理解的交叉熵
    label_mask = label_target.eq(0)  # 不是头节点的置为1
    label_target = label_target.masked_fill(label_mask, -1)  # ignore_index=-1
    label_scores = label_scores.contiguous().view(-1, len(vocabs['rel']))  # 根据140重新计算维度
    rel_loss = crit_rel(label_scores, label_target.view(
        -1))  # batch len len label_num <> batch len len  target 是类别class 的索引([0,C−1], C 是类别classes 总数.)

    word_num = torch.sum(torch.eq(word_mask, 0)).item()  # 此前填充的已经变为0
    loss = head_loss + rel_loss
    loss /= word_num  # number of words

    return loss
