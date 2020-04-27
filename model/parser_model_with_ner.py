#    Author:  a101269
#    Date  :  2020/3/9

import torch
import torch.nn as nn
from modules.parser_encoder import Parser_encoder
from modules.biaffine import DeepBiaffineScorer
from sequence_tagging.crf import CRF
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

        biaff_input_size = args.bert_dim
        if args.use_pos or args.use_knowledge:
            biaff_input_size = 0
            self.trans_bert_dim_layer = nn.Linear(args.bert_dim, args.bert_trans_dim)
            biaff_input_size += args.bert_trans_dim
        if args.use_knowledge:
            self.know_emb = nn.Embedding(len(vocabs['bio']), args.knowledge_dim, padding_idx=0)
            biaff_input_size += self.args.knowledge_dim

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

        self.head_biaffine = DeepBiaffineScorer(biaff_input_size, biaff_input_size,
                                                args.biaffine_hidden_dim, 1, pairwise=True,
                                                dropout=args.biaffine_dropout)
        self.relaton_biaffine = DeepBiaffineScorer(biaff_input_size, biaff_input_size,
                                                   args.biaffine_hidden_dim, len(vocabs['rel']), pairwise=True,
                                                   dropout=args.biaffine_dropout)
        # mudules for sequence tagging
        self.tagset_size = len(self.vocabs['bio'])
        self.crf = CRF(target_size=self.tagset_size, average_batch=True, use_cuda=self.args.use_cuda)
        self.ner_linear = nn.Linear(args.bert_dim, self.tagset_size)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, pos_ids=None,
                knowledge_feature=None, boundary_ids=None, bio_ids=None):#,knowledge_adjoin_matrix=None
        embeddings, max_len_of_batch,tail_mask = self.encoder(input_ids, token_type_ids=token_type_ids,
                                                    attention_mask=attention_mask,
                                                    boundary_ids=boundary_ids,
                                                    knowledge_feature=knowledge_feature)
        embeddings=self.dropout(embeddings)
        # 序列标注
        ner_scores = self.ner_linear(embeddings)
        tail_mask = torch.eq(tail_mask, False)

        _, tag_seq = self.crf._viterbi_decode(ner_scores, tail_mask)
        # pad_zero = torch.cuda.LongTensor(tag_seq.size()[0], max_len_of_batch) if torch.cuda.is_available() else torch.LongTensor(tag_seq.size()[0], max_len_of_batch)
        # pad_zero *= 0
        # tag_seq = torch.where(tag_seq >=4, tag_seq, pad_zero)# 满足condition，原样返回x的值，不满足才填充

        # 语义依存
        if self.args.use_knowledge or self.args.use_pos:
            embeddings = self.trans_bert_dim_layer(embeddings)

        if self.args.use_pos:
            pos_emb = self.pos_emb(pos_ids[:, :max_len_of_batch])
            embeddings = torch.cat((embeddings, pos_emb), 2)

        # cat  SEP  SEP 必要
        if self.args.use_knowledge:
            knowledge_feature =tag_seq # bio_ids[:, :max_len_of_batch]
            if self.args.use_gat:  # 使用GAT
                know_label_map = self.know_emb(self.knowledge_label)
                know_label_map = self.graph_attention_networks(know_label_map, self.relation_matrix)  # 20 * 128
                know_emb_select=nn.Embedding.from_pretrained(know_label_map)
                know_emb=know_emb_select(knowledge_feature)
                # knowledge_adjoin_matrix=knowledge_adjoin_matrix[:,:max_len_of_batch, :max_len_of_batch]
            else:  # 拼接
                know_emb = self.know_emb(knowledge_feature)
            embeddings = torch.cat((embeddings, know_emb), 2)

        # embeddings=self.dropout(embeddings)
        if self.args.use_transformer:
            embeddings = self.transformer_encoder(embeddings)  # torch.Size([20, 65, 768])
        head_scores = self.head_biaffine(embeddings, embeddings).squeeze(3)
        label_scores = self.relaton_biaffine(embeddings, embeddings)

        return head_scores, label_scores, max_len_of_batch,ner_scores

    # def get_output_score(self, sentence, attention_mask=None,csr_out=None):
    #     batch_size = sentence.size(0)
    #     seq_length = sentence.size(1)
    #     embeds, _ = self.word_embeds(sentence, attention_mask=attention_mask)
    #     # if embeds.is_cuda:
    #     #     hidden = (i.cuda() for i in hidden)
    #     embeds=self.dropout(embeds)
    #     if csr_out:
    #         embeds=torch.cat((embeds,csr_out))
    #     l_out = self.liner(embeds)
    #     lstm_feats = l_out.contiguous().view(batch_size, seq_length, -1)
    #     return lstm_feats
    #
    # def do_ner(self, sentence, masks):
    #     lstm_feats = self.get_output_score(sentence)
    #     scores, tag_seq = self.crf._viterbi_decode(lstm_feats, masks.byte())
    #     return tag_seq
    #
    # def neg_log_likelihood_loss(self, sentence, mask, tags,csr_out=None):
    #     lstm_feats = self.get_output_score(sentence,csr_out=csr_out)
    #     loss_value = self.crf.neg_log_likelihood_loss(lstm_feats, mask, tags)
    #     batch_size = lstm_feats.size(0)
    #     loss_value /= float(batch_size)
    #     return loss_value
