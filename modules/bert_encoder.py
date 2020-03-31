#    Author:  a101269
#    Date  :  2020/3/9
import torch.nn as nn
from transformers import BertModel

class Bert_encoder(nn.Module):
    def __init__(self,model_path):
        super(Bert_encoder, self).__init__()
        # modelConfig = BertConfig.from_pretrained(model_path)
        # model = BertModel.from_pretrained(model_path,config=modelConfig)
        self.bert_model = BertModel.from_pretrained(model_path)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, head_mask=None):
        bert_outputs = self.bert_model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                            head_mask=head_mask)
        bert_output = bert_outputs[0]

        return bert_output
