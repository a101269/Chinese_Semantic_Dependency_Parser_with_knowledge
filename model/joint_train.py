#    Author:  a101269
#    Date  :  2020/3/9
import torch
import torch.nn as nn
from utils.utils import logger, seed_everything
from model.criterion import criterion
from model.optimizer import get_optimizer
from torch.nn.utils import clip_grad_norm_
from modules.sdp_decoder import sdp_decoder, parse_semgraph
import model.cal_las_uas as sdp_scorer
from tensorboardX import SummaryWriter


def unpack_batch(batch, use_cuda=False):
    """ Unpack a batch from the data loader. """
    input_ids = batch[0]
    input_mask = batch[1]
    segment_ids = batch[2]
    boundary_ids = batch[3]
    pos_ids = batch[4]
    rel_ids = batch[5]
    knowledge_feature = batch[6]
    bio_ids = batch[1]
    # knowledge_adjoin_matrix = batch[7]
    # know_segment_ids = batch[6]
    # know_input_ids = batch[7]
    # know_input_mask = batch[8]
    # knowledge_feature = (batch[6], batch[7], batch[8])

    return input_ids, input_mask, segment_ids, boundary_ids, pos_ids, rel_ids, knowledge_feature,bio_ids#,knowledge_adjoin_matrix


class Trainer(object):
    def __init__(self, args, model, batch_num=None):
        self.model = model
        self.use_cuda = args.use_cuda
        self.device = args.device
        self.fp16 = args.fp16
        self.args = args

        self.global_step = 0
        self.batch_num = batch_num
        self.optimizer, self.lr_scheduler,self.optimizer2,self.lr_scheduler2 = get_optimizer(args, batch_num, self.model)
        if self.use_cuda:
            self.model.cuda()
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, self.optimizer,self.optimizer2 = amp.initialize(model, self.optimizer, opt_level=args.fp16_opt_level)

    def train(self, train_dataloader, dev_dataloader=None, dev_conllu_file=None):
        summary_writer = SummaryWriter('board_log')
        seed_everything(self.args.seed)
        best_las = 0
        best_uas = 0
        self.args.eval_interval = int(self.batch_num / 2)
        logger.info(f"eval_interval:{self.args.eval_interval}")
        for epoch in range(self.args.epochs):
            if best_las > 0.829:
                self.args.eval_interval = 300
                # logger.info(f"eval_interval:{self.args.eval_interval}")
            for step, batch in enumerate(train_dataloader):
                self.model.train()
                self.global_step += 1
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, boundary_ids, pos_ids, rel_ids, knowledge_feature,bio_ids= unpack_batch(batch, self.use_cuda)
                if self.global_step==0:
                    dummy_input=(input_ids, segment_ids,input_mask,pos_ids,boundary_ids,knowledge_feature)
                    summary_writer.add_graph(self.model, (dummy_input,))
                head_scores, label_scores, max_len_of_batch, ner_scores= self.model(input_ids, token_type_ids=segment_ids,
                                                                         attention_mask=input_mask, pos_ids=pos_ids,
                                                                         boundary_ids=boundary_ids,
                                                                         knowledge_feature=knowledge_feature,
                                                                         bio_ids=bio_ids)

                label_target = rel_ids[:, :max_len_of_batch, :max_len_of_batch]
                head_target = label_target.ge(2).type_as(rel_ids)  # VOCAB_PREFIX = [PAD, UNK, ROOT],root索引为2

                tails = boundary_ids[:, :max_len_of_batch]
                word_mask = torch.eq(tails, 0)  # 填充的变为1
                loss = criterion(head_scores, label_scores, head_target, label_target, word_mask, max_len_of_batch,
                                 self.model.vocabs)
                # if self.global_step%3!=0:
                pad_zero = torch.cuda.LongTensor(bio_ids.size()[0],max_len_of_batch) if torch.cuda.is_available() else torch.LongTensor(bio_ids.size()[0], max_len_of_batch)
                pad_zero *= 0
                bio_ids= bio_ids[:, :max_len_of_batch]
                bio_ids = torch.where(bio_ids >= 5, bio_ids, pad_zero)
                ner_loss = self.model.crf.neg_log_likelihood_loss(ner_scores, input_mask[:, :max_len_of_batch], bio_ids)
                ner_loss /= float(self.args.batch_size)
                summary_writer.add_scalar('ner_loss', ner_loss, self.global_step)
                if ner_loss>0.6:
                    ner_loss.backward(retain_graph=True)
                    self.optimizer2.step()
                    self.lr_scheduler2.step()
                    self.optimizer2.zero_grad()
                # logger.info('loss   %s', loss)
                summary_writer.add_scalar('parser_loss',loss, self.global_step)

                if self.fp16:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                    clip_grad_norm_(amp.master_params(self.optimizer), self.args.grad_clip_max_norm)
                else:
                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), self.args.grad_clip_max_norm)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()


                if self.global_step % self.args.eval_interval == 0 or self.global_step==self.batch_num*self.args.epochs:
                    LAS, UAS = self.predict(dev_dataloader, dev_conllu_file)
                    if LAS > best_las:
                        best_las = LAS
                        best_uas = UAS
                        self.save_model()
                        logger.warning(
                            f"epoch{epoch+1}, step:{self.global_step}-----LAS:{best_las:.4f},UAS:{UAS:.4f},loss {loss.item():.4f}")
                    else:
                        logger.info(f"LAS ,UAS in epoch{epoch+1},step{step+1}:{LAS:.4f},{UAS:.4f}")

                    summary_writer.add_scalar('LAS', LAS, self.global_step)
                    summary_writer.add_scalar('UAS', UAS, self.global_step)
                    for i, param_group in enumerate(self.optimizer.param_groups):
                        summary_writer.add_scalar(f'lr/group_{i}', param_group['lr'], self.global_step)

        summary_writer.close()
        logger.warning(f"Result in Dev set:  LAS:{best_las:.4f},UAS:{best_uas:.4f}")

    def predict(self, dev_dataloader, dev_conllu_file):
        predictions = []
        self.model.eval()

        for step, batch in enumerate(dev_dataloader):
            with torch.no_grad():
                preds_batch = []
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, boundary_ids, pos_ids, rel_ids, knowledge_feature,bio_ids= unpack_batch(
                    batch,self.use_cuda)

                head_scores, label_scores, max_len_of_batch, ner_scores= self.model(input_ids, token_type_ids=segment_ids,
                                                                         attention_mask=input_mask, pos_ids=pos_ids,
                                                                         boundary_ids=boundary_ids,
                                                                         knowledge_feature=knowledge_feature,
                                                                         bio_ids=bio_ids)
                batch_size = head_scores.size(0)
                tails = boundary_ids[:, :max_len_of_batch]

                word_mask = torch.eq(tails, 0)  # 填充的变为1
                weights = torch.ones(batch_size, max_len_of_batch, max_len_of_batch, dtype=torch.float,
                                     device=self.device)
                weights = weights.masked_fill(word_mask.unsqueeze(1), 0)  # 将填充的置0
                weights = weights.masked_fill(word_mask.unsqueeze(2), 0)
                weights = weights.unsqueeze(3)
                # print(unlabeled_scores[0])  # 跑模型，评估时输出的大长串
                head_probs = torch.sigmoid(head_scores).unsqueeze(3)  # [100, 44, 44, 1]
                label_probs = torch.softmax(label_scores, dim=3)  # [100, 44, 44, 144]
                semgraph_probs = head_probs * label_probs * weights  # 因此解码时并非先解码弧，再单独判断标签，而是弧与标签的联合概率
                preds_batch.append(semgraph_probs.detach().cpu().numpy())  # detach（），截断反向传播的梯度流

                tail_mask = tails != 0
                sentlens = torch.sum(tail_mask, 1).cpu().tolist()
                semgraph = sdp_decoder(preds_batch[0], sentlens)
                sents = parse_semgraph(semgraph, sentlens)
                pred_sents = self.model.vocabs['rel'].parse_to_sent_batch(sents)

            predictions += pred_sents

        dev_conllu_file.set(['deps'], [dep for sent in predictions for dep in sent])
        dev_conllu_file.write_conll(self.args.eval_temp_file)
        UAS, LAS = sdp_scorer.score(self.args.eval_temp_file, self.args.gold_file)

        return LAS, UAS

    def save_model(self):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        # model_to_save.save_pretrained(str(self.args.saved_model_path))
        saved_model_file = self.args.saved_model_path + '/pytorch_model.bin'
        torch.save(model_to_save.state_dict(), saved_model_file)
        output_config_file = self.args.saved_model_path + '/config.json'
        with open(str(output_config_file), 'w') as f:
            f.write(model_to_save.encoder.bert_model.config.to_json_string())
