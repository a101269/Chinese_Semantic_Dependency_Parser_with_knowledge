#    Author:  a101269
#    Date  :  2020/3/9

import torch
import torch.nn as nn
from modules.parser_encoder import Parser_encoder
from modules.biaffine import DeepBiaffineScorer
from modules.graph_attention_model import GAT as GAT_model
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
            if args.use_knowledge:
                self.seg_emb_dim=4
            else:
                self.seg_emb_dim = 8
            self.seg_emb= torch.cuda.FloatTensor(self.seg_emb_dim) if torch.cuda.is_available() else torch.FloatTensor(self.seg_emb_dim)
            self.seg_emb*=0
            biaff_input_size += self.seg_emb_dim
        if args.use_knowledge:
            self.know_emb = nn.Embedding(len(vocabs['know']), args.knowledge_dim, padding_idx=0)

            if args.use_gat:
                self.knowledge_label = torch.arange(0, self.know_label_num, 1,dtype=torch.long).to(self.args.device)
                self.relation_matrix = torch.zeros([self.know_label_num, self.know_label_num])
                for label, relation_labels in know_relaton.items():
                    for rl in relation_labels:
                        self.relation_matrix[self.vocabs['know'].unit2id(label), self.vocabs['know'].unit2id(rl)] = 1
                        self.relation_matrix=self.relation_matrix.to(self.args.device)
                self.graph_attention_networks =GAT_model(nfeat=self.args.knowledge_dim, nhid=args.gat_hidden,
                                                    out_dim=self.args.knowledge_dim, dropout=args.gat_dropout,
                                                    nheads=args.gat_heads, alpha=args.gat_alpha)

                # self.graph_attention_networks=GAT(nfeat=self.args.knowledge_dim+args.bert_trans_dim, nhid=args.gat_hidden, out_dim=1, dropout=args.gat_dropout,
                #                                     nheads=args.gat_heads, alpha=args.gat_alpha, layer=1)
            biaff_input_size += self.args.knowledge_dim
            biaff_input_size +=self.seg_emb_dim


        if args.use_pos:
            self.pos_emb = nn.Embedding(len(vocabs['pos']), args.pos_dim, padding_idx=0)
            biaff_input_size += self.args.pos_dim

        if args.use_transformer:
            transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=biaff_input_size, nhead=8,
                                                                   dim_feedforward=2048,
                                                                   dropout=args.transformer_dropout,
                                                                   activation='gelu')
            self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer,
                                                             num_layers=args.transformer_layer)

            # torch.nn.Transformer(d_model=768, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
            # dropout=0.1, activation='relu', custom_encoder=None, custom_decoder=None)

            # transformer_encoder_layer_pre = nn.TransformerEncoderLayer(d_model=biaff_input_size, nhead=6,
            #                                                        dim_feedforward=1536,
            #                                                        dropout=args.transformer_dropout,
            #                                                        activation='gelu')
            # self.transformer_encoder_pre = nn.TransformerEncoder(transformer_encoder_layer_pre,
            #                                                  num_layers=1)


        self.head_biaffine = DeepBiaffineScorer(biaff_input_size, biaff_input_size,
                                                args.biaffine_hidden_dim, 1, pairwise=True,
                                                dropout=args.biaffine_dropout)
        self.relaton_biaffine = DeepBiaffineScorer(biaff_input_size, biaff_input_size,
                                                   args.biaffine_hidden_dim, len(vocabs['rel']), pairwise=True,
                                                   dropout=args.biaffine_dropout)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, pos_ids=None,
                knowledge_feature=None, boundary_ids=None):#,knowledge_adjoin_matrix=None
        embeddings, max_len_of_batch = self.encoder(input_ids, token_type_ids=token_type_ids,
                                                    attention_mask=attention_mask,
                                                    boundary_ids=boundary_ids,
                                                    knowledge_feature=knowledge_feature)
        embeddings=self.dropout(embeddings)
        if self.args.use_knowledge or self.args.use_pos:
            embeddings = self.trans_bert_dim_layer(embeddings)

        if self.args.use_pos:
            pos_emb = self.pos_emb(pos_ids[:, :max_len_of_batch])
            seg_emb = self.seg_emb.unsqueeze(0).unsqueeze(0).expand(embeddings.size(0),max_len_of_batch,  self.seg_emb_dim)
            embeddings = torch.cat((embeddings, seg_emb ), 2)
            embeddings = torch.cat((embeddings, pos_emb), 2)

        # cat  SEP  SEP 必要
        if self.args.use_knowledge:
            knowledge_feature = knowledge_feature[:, :max_len_of_batch]
            if self.args.use_gat:  # 使用GAT
                know_label_map = self.know_emb(self.knowledge_label)
                know_label_map = self.graph_attention_networks(know_label_map, self.relation_matrix)  # 20 * 128
                know_emb_select=nn.Embedding.from_pretrained(know_label_map)
                know_emb=know_emb_select(knowledge_feature)
                # knowledge_adjoin_matrix=knowledge_adjoin_matrix[:,:max_len_of_batch, :max_len_of_batch]
            else:  # 拼接
                know_emb = self.know_emb(knowledge_feature)
            seg_emb = self.seg_emb.unsqueeze(0).unsqueeze(0).expand(embeddings.size(0),max_len_of_batch,  self.seg_emb_dim)
            embeddings = torch.cat((embeddings, seg_emb ), 2)
            embeddings = torch.cat((embeddings, know_emb), 2)

        # embeddings=self.dropout(embeddings)
        if self.args.use_transformer:
            embeddings = self.transformer_encoder(embeddings)  # torch.Size([20, 65, 768])
        head_scores = self.head_biaffine(embeddings, embeddings).squeeze(3)
        label_scores = self.relaton_biaffine(embeddings, embeddings)

        return head_scores, label_scores, max_len_of_batch
