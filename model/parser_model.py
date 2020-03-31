#    Author:  a101269
#    Date  :  2020/3/9

import torch
import torch.nn as nn
from modules.parser_encoder import Parser_encoder
from modules.biaffine import DeepBiaffineScorer
from utils.utils import logger


class Parser_model(nn.Module):
    def __init__(self, args, vocabs):
        super(Parser_model, self).__init__()
        self.args = args
        self.device = args.device
        self.vocabs = vocabs
        self.unsaved_modules = []
        self.encoder = Parser_encoder(args.pretrain_model_path)
        self.dropout = nn.Dropout(args.bert_dropout)

        biaff_input_size = args.bert_dim
        if args.use_pos or args.use_knowledge:
            biaff_input_size = 0
            self.trans_bert_dim_layer = nn.Linear(args.bert_dim, args.bert_trans_dim)
            biaff_input_size += args.bert_trans_dim
        if args.use_pos:
            self.pos_emb = nn.Embedding(len(vocabs['pos']), args.pos_dim, padding_idx=0)
            biaff_input_size += self.args.pos_dim
        if args.use_knowledge:
            # self.trans_knowledge_dim_layer = nn.Linear(args.bert_dim, args.knowledge_dim)
            self.know_emb = nn.Embedding(len(vocabs['know']),  args.knowledge_dim, padding_idx=0)
            biaff_input_size += self.args.knowledge_dim

        if args.use_transformer:
            transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=biaff_input_size, nhead=8,
                                                                   dim_feedforward=2048,
                                                                   dropout=args.transformer_dropout,
                                                                   activation='gelu')
            self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer,
                                                             num_layers=args.transformer_layer)
            # torch.nn.Transformer(d_model=768, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
            # dropout=0.1, activation='relu', custom_encoder=None, custom_decoder=None)

        self.head_biaffine = DeepBiaffineScorer(biaff_input_size, biaff_input_size,
                                                args.biaffine_hidden_dim, 1, pairwise=True,
                                                dropout=args.biaffine_dropout)
        self.relaton_biaffine = DeepBiaffineScorer(biaff_input_size, biaff_input_size,
                                                   args.biaffine_hidden_dim, len(vocabs['rel']), pairwise=True,
                                                   dropout=args.biaffine_dropout)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, pos_ids=None,
                knowledge_feature=None, boundary_ids=None):
        bert_output, max_len_of_batch = self.encoder(input_ids, token_type_ids=token_type_ids,
                                                                   attention_mask=attention_mask,
                                                                   boundary_ids=boundary_ids,
                                                                   knowledge_feature=knowledge_feature)
        # logger.warning('bert_output shape:%s', bert_output.shape)
        # logger.warning('know_encode shape:%s', knowledge.shape)

        bert_output = self.dropout(bert_output)
        bert_output = self.trans_bert_dim_layer(bert_output)

        if self.args.use_pos:
            pos_emb = self.pos_emb(pos_ids[:, :max_len_of_batch])
            embeddings = torch.cat((bert_output, pos_emb), 2)
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! cat  SEP  SEP
        if self.args.use_knowledge:
            know_emb=self.know_emb(knowledge_feature[:, :max_len_of_batch])
            # knowledge= self.trans_knowledge_dim_layer(knowledge)
            embeddings = torch.cat((embeddings, know_emb), 2)

        if self.args.use_transformer:
            embeddings = self.transformer_encoder(embeddings)  # torch.Size([20, 65, 768])

        head_scores = self.head_biaffine(embeddings, embeddings).squeeze(3)
        label_scores = self.relaton_biaffine(embeddings, embeddings)

        return head_scores, label_scores, max_len_of_batch
