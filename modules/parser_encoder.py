#    Author:  a101269
#    Date  :  2020/3/9

import torch
import torch.nn as nn
from transformers import BertModel
from utils.utils import logger


class Parser_encoder(nn.Module):

    def __init__(self, model_path):
        super(Parser_encoder, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_path, output_hidden_states=False, output_attentions=False)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, pos_ids=None,
                knowledge_feature=None, boundary_ids=None):

        sent_len = torch.sum(attention_mask, 1).cpu().tolist()
        max_len_of_batch = max(sent_len)

        bert_outputs = self.bert_model(input_ids[:max_len_of_batch], token_type_ids=token_type_ids[:max_len_of_batch], attention_mask=attention_mask[:max_len_of_batch])
        # sequence_output, pooled_output, (hidden_states), (attentions)

        bert_output = bert_outputs[0]  # torch.Size([4, 256, 768]

        # 后n层加和
        # all_layers_hidden_states = bert_outputs[2]
        # last_four_hidden_states = torch.stack(all_layers_hidden_states[-4:])
        # bert_output = torch.sum(last_four_hidden_states, 0)

        # [4, 256,1] expand 单个维度扩大为更大的尺寸,因此维度是1的维度将扩大
        # logger.info('bert_output shape:%s',bert_output.shape)
        # logger.info('boundary_ids shape:%s',boundary_ids.shape)
        # boundary_ids = boundary_ids[:, :max_len_of_batch]

        tail_index = boundary_ids.unsqueeze(2).expand(boundary_ids.size(0), boundary_ids.size(1), bert_output.size(2))

        # 获得词尾向量
        bert_tail_cls = bert_output.gather(1, tail_index)

        tail_mask = boundary_ids != 0
        logger.debug(tail_mask)
        sent_len = torch.sum(tail_mask, 1).cpu().tolist()
        max_len_of_batch = max(sent_len)
        # logger.debug(sent_len)
        # logger.debug('max_len_of_batch:%s', max_len_of_batch)

        # 裁剪多余的填充
        bert_tail = bert_tail_cls[:, :max_len_of_batch]
        # logger.debug('bert_tail shape:%s', bert_tail.shape)
        # logger.debug('bert_tail shape:%s', bert_tail)

        boundary_ids = boundary_ids[:, :max_len_of_batch]
        tail_mask = torch.eq(boundary_ids, 0)
        bert_tail = bert_tail.masked_fill(tail_mask.unsqueeze(2), 0)

        return bert_tail, max_len_of_batch ,bert_tail
        # logger.debug('clip_bert_tail shape:%s', bert_tail.shape)
        # logger.debug('clip_bert_tail shape:%s', bert_tail)
        # know_segment_ids, know_input_ids, know_input_mask = knowledge_feature
        # know_encodes = []

        # self.bert_model.eval()
        # for i, kin_id in enumerate(know_input_ids):
        #     know_bert_out = self.bert_model(kin_id[:max_len_of_batch], token_type_ids=know_segment_ids[i][:max_len_of_batch],
        #                                     attention_mask=know_input_mask[i][:max_len_of_batch])
        #     know_encode = know_bert_out[1].detach()  # [256, 768]
        #     know_encodes.append(know_encode)
        #     # del know_bert_out
        # if not self.do_test:
        #     self.bert_model.train()
        #
        # know_encodes = torch.stack(know_encodes)
        # logger.warning('know_encode shape:%s', know_encodes.shape)
        # logger.info('know_encode :%s',know_encode )
        # know_encode [4, 256,20,768] [4, 256, 768]

        # return bert_tail, know_encodes, max_len_of_batch
