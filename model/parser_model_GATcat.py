#    Author:  a101269
#    Date  :  2020/3/9

import torch
import torch.nn as nn
from modules.parser_encoder import Parser_encoder
from modules.biaffine import DeepBiaffineScorer
# from modules.graph_attention_model import GAT
from modules.graph_attention_networks import GAT
from modules.adj_graph import know_relaton, creat_adjoin_matrix
from utils.utils import logger


class Parser_model(nn.Module):
    def __init__(self, args, vocabs):
        super(Parser_model, self).__init__()
        self.args = args
        self.device = args.device
        self.vocabs = vocabs
        self.know_label_num = len(self.vocabs['know'])
        self.unsaved_modules = []
        self.encoder = Parser_encoder(args.pretrain_model_path)
        self.dropout = nn.Dropout(args.bert_dropout)
        # self.adj_graph=adj_graph()

        biaff_input_size = args.bert_dim
        if args.use_pos or args.use_knowledge:
            biaff_input_size = 0
            self.trans_bert_dim_layer = nn.Linear(args.bert_dim, args.bert_trans_dim)
            biaff_input_size += args.bert_trans_dim
        if args.use_pos:
            self.pos_emb = nn.Embedding(len(vocabs['pos']), args.pos_dim, padding_idx=0)
            biaff_input_size += self.args.pos_dim
        if args.use_knowledge:
            self.know_emb = nn.Embedding(len(vocabs['know']), args.knowledge_dim, padding_idx=0)

            if args.use_gat:
                self.knowledge_label = torch.arange(0, self.know_label_num, 1,dtype=torch.long).to(self.args.device)
                # self.relation_matrix = torch.zeros([self.know_label_num, self.know_label_num])
                # for label, relation_labels in know_relaton.items():
                #     for rl in relation_labels:
                #         self.relation_matrix[self.vocabs['know'].unit2id(label), self.vocabs['know'].unit2id(rl)] = 1
                #         self.relation_matrix=self.relation_matrix.to(self.args.device)
                # self.graph_attention_networks = GAT(nfeat=self.args.knowledge_dim, nhid=args.gat_hidden,
                #                                     out_dim=self.args.knowledge_dim, dropout=args.gat_dropout,
                #                                     nheads=args.gat_heads, alpha=args.gat_alpha)

                self.graph_attention_networks=GAT(nfeat=self.args.knowledge_dim, nhid=args.gat_hidden, out_dim=self.args.knowledge_dim, dropout=args.gat_dropout,
                                                    nheads=args.gat_heads, alpha=args.gat_alpha, layer=1)
            # biaff_input_size += self.args.knowledge_dim

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
        embeddings, max_len_of_batch = self.encoder(input_ids, token_type_ids=token_type_ids,
                                                    attention_mask=attention_mask,
                                                    boundary_ids=boundary_ids,
                                                    knowledge_feature=knowledge_feature)

        # embeddings = self.dropout(embeddings)
        if self.args.use_knowledge or self.args.use_pos:
            embeddings = self.trans_bert_dim_layer(embeddings)

        if self.args.use_pos:
            pos_emb = self.pos_emb(pos_ids[:, :max_len_of_batch])
            embeddings = torch.cat((embeddings, pos_emb), 2)
        # cat  SEP  SEP 必要
        if self.args.use_knowledge:
            knowledge_feature = knowledge_feature[:, :max_len_of_batch]

            # knowledge= self.trans_knowledge_dim_layer(knowledge)
            if self.args.use_gat:  # 使用GAT
                know_emb_map = self.know_emb(self.knowledge_label)
                # adj_matr =  self.adj_graph.creat_adj_matrix(max_len_of_batch, self.know_label_num, knowledge_feature=knowledge_feature,
                #                             know_relaton=self.relation_matrix)
                know_emb_map = self.graph_attention_networks(know_emb_map ,self.relation_matrix)  # 20 * 128

                for i,idxes in enumerate(knowledge_feature):
                    if i==0:
                        know_emb = torch.index_select(know_emb_map, 0, idxes).unsqueeze(0)
                    # logger.warning('know_emb shape:%s',know_emb.shape) # torch.Size([76, 128])
                    if i > 0:
                        know_emb=torch.cat((know_emb,torch.index_select(know_emb_map, 0, idxes).unsqueeze(0)))

                embeddings = torch.cat((embeddings, know_emb), 2)
            else:  # 拼接
                know_emb = self.know_emb(knowledge_feature)
                embeddings = torch.cat((embeddings, know_emb), 2)

        if self.args.use_transformer:
            embeddings = self.transformer_encoder(embeddings)  # torch.Size([20, 65, 768])

        head_scores = self.head_biaffine(embeddings, embeddings).squeeze(3)
        label_scores = self.relaton_biaffine(embeddings, embeddings)

        return head_scores, label_scores, max_len_of_batch
