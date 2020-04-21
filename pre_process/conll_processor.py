#    Author:  a101269
#    Date  :  2020/3/5

import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer
from pre_process import conll
# from pre_process.adj_graph import creat_sent_adj
from utils import utils
from utils.utils import logger


def make_labeltarget(arcs, max_seq_length):
    graphs = [[0] * max_seq_length for _ in range(max_seq_length)]
    word_idx = 1
    for word in arcs:
        for arc in word:
            # print(sent_idx, word_idx, arc)
            head_idx = arc[0]
            rel_idx = arc[1]
            graphs[word_idx][head_idx] = rel_idx
        word_idx += 1
    return graphs


class InputExample(object):
    def __init__(self, guid, text_a, pos=None, knowledge=None, boundary=None, arcs=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.pos = pos
        self.knowledge = knowledge
        self.boundary = boundary
        self.arcs = arcs


class InputFeature(object):
    '''
    A single set of features of data.
    '''

    def __init__(self, input_ids, input_mask, segment_ids, boundary_ids, pos_ids, input_len, rel_ids,
                 know_ids=None, know_adj_mar=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.boundary_ids = boundary_ids
        self.pos_ids = pos_ids
        self.input_len = input_len
        self.rel_ids = rel_ids
        self.know_ids = know_ids
        self.know_adj_mar = know_adj_mar

class ConllProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, args, vocabs):
        # self.tokenizer = BertTokenizer(args.vocab_path, args.do_lower_case)
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path, do_lower_case=args.do_lower_case,
                                                       additional_special_tokens=['[unused1]', '[unused2]'])
        self.vocabs = vocabs

    def get_train(self, data_file):
        """Gets a collection of `InputExample`s for the train set."""
        return self.read_data(data_file)

    def get_dev(self, data_file):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.read_data(data_file)

    def get_test(self, data_file):
        return self.read_data(data_file)

    @classmethod
    def read_data(self, data_file):
        conll_file = conll.CoNLLFile(data_file)
        data = conll_file.get(['word', 'upos', 'deps', 'misc'], as_sentences=True)  # 没有头节点原因 5:rAgt|12:rAgt
        return conll_file, data

    def truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def create_examples(self, lines, example_type, cached_examples_file):
        '''
        Creates examples for data
        '''
        examples = []
        for i, line in enumerate(lines):
            guid = '%s-%d' % (example_type, i)
            text_a = ['[unused1]']  # root
            pos = ['<ROOT>']
            arcs = []  # root时略过
            knowledge = []
            boundary = [0]  # 单词边界，词尾索引 [cls],root

            if len(line[0]) == 4 and line[0][-2] != '_':
                knowledge.append('<ROOT>')
            for word in line:
                head_rel = []
                text_a.append(word[0])  # [ ['外公', 'NN', '7:Agt'], ['走', 'VV', '2:dCont'], ['吗', 'SP', '7:mTone'] ]
                pos.append(word[1])

                if utils.has_ens_num(word[0]):
                    boundary.append(boundary[-1] + len(self.tokenizer.tokenize(word[0])))
                else:
                    boundary.append((boundary[-1] + len(word[0])))

                hrs = word[2].split('|')
                for hr in hrs:
                    head, rel = hr.split(':')
                    dep_rel_idx = self.vocabs['rel'].unit2id(rel)
                    head_rel.append((int(head), dep_rel_idx))
                arcs.append(head_rel)
                if knowledge:
                    knowledge.append(word[3])  # ['外公', 'NN', '7:Agt', '人物，亲属']
            if i == len(lines) - 1:
                print(guid, pos, knowledge, boundary, arcs)
            text_a = ''.join(text_a)
            example = InputExample(guid=guid, text_a=text_a, pos=pos, knowledge=knowledge, boundary=boundary, arcs=arcs)
            examples.append(example)
        torch.save(examples, cached_examples_file)
        logger.info(f'Done: cached examples to {cached_examples_file}')

        return examples

    def create_features(self, examples, cached_features_file, max_seq_len):
        '''
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        '''

        features = []
        for ex_id, example in enumerate(examples):
            tokens_a = self.tokenizer.tokenize(example.text_a)
            if len(tokens_a) > max_seq_len - 2:
                # logger.warning(len(tokens_a))
                logger.info(tokens_a)
                continue
            # Account for [CLS] and [SEP] with '-2'
            # if len(tokens_a) > max_seq_len - 2:
            #     logger.warning('max_seq_len:256!!!!')

            tokens = ['[CLS]'] + tokens_a + ['[SEP]']
            segment_ids = [0] * len(tokens)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            input_mask = [1] * len(input_ids)
            padding = [0] * (max_seq_len - len(input_ids))
            input_len = len(input_ids)
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            assert len(input_ids) == max_seq_len
            assert len(input_mask) == max_seq_len
            assert len(segment_ids) == max_seq_len

            pos_a = example.pos
            poses = pos_a
            pos_ids = self.vocabs['pos'].map(poses)

            padding = [0] * (max_seq_len - len(pos_ids))
            pos_ids += padding
            # assert len(pos_ids) == max_seq_len-2

            boundary_ids = [b + 1 for b in example.boundary]  #
            # for b in boundary_ids:  # ram', '##bus  1111
            #     if b>99:
            #         print('max',max_seq_len)
            #         print(b)
            #         print(len((tokens)))
            #         print(tokens)
            #         print(boundary_ids)
            #         break
            padding = [0] * (max_seq_len - len(boundary_ids))  # 0 的话会与CLS 位置搞混，不会
            boundary_ids += padding

            arcs = example.arcs
            arcs_ids = make_labeltarget(arcs, max_seq_len)

            know_ids = None
            if example.knowledge:
                know_a = example.knowledge
                knowes = know_a
                know_ids = self.vocabs['know'].map(knowes)
                know_ids += padding
                assert len(know_ids) == len(pos_ids)
                # know_adj_mar= creat_sent_adj(max_seq_len, none_label=self.vocabs['rel'].unit2id('_'), knowledge_feature=know_ids)

            if ex_id < 2:
                logger.info("*** Example ***")
                logger.info(f"guid: {example.guid}" % ())
                logger.info(f"tokens: {' '.join([str(x) for x in tokens])}")
                logger.info(f"input_ids: {' '.join([str(x) for x in input_ids])}")
                logger.info(f"input_mask: {' '.join([str(x) for x in input_mask])}")
                logger.info(f"segment_ids: {' '.join([str(x) for x in segment_ids])}")
                logger.info(f"arc_ids: {' '.join([str(x) for x in example.arcs])}")
                logger.info(f"pos_ids: {' '.join([str(x) for x in pos_ids])}")
                logger.info(f"boundary_ids: {' '.join([str(x) for x in boundary_ids])}")
                if know_ids:
                    logger.info(f"know_ids: {' '.join([str(x) for x in know_ids])}")

            feature = InputFeature(input_ids=input_ids,
                                   input_mask=input_mask,
                                   segment_ids=segment_ids,
                                   boundary_ids=boundary_ids,
                                   pos_ids=pos_ids,
                                   rel_ids=arcs_ids,
                                   input_len=input_len,
                                   know_ids=know_ids,

                                   # knowledge_segment_ids=knowledge_segment_ids,
                                   # knowledge_input_ids=knowledge_input_ids,
                                   # knowledge_input_mask=knowledge_input_mask,
                                   # know_ids=know_ids
                                   )

            features.append(feature)
        torch.save(features, cached_features_file)
        logger.info("Saved features into cached file %s", cached_features_file)
        return features

    def create_dataset(self, features, is_sorted=False):
        # Convert to Tensors and build dataset
        if is_sorted:
            features = sorted(features, key=lambda x: x.input_len, reverse=True)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)  # torch.Size([534, 256])
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        # logger.warning(all_input_ids.shape)
        all_boundary_ids = torch.tensor([f.boundary_ids for f in features], dtype=torch.long)
        all_pos_ids = torch.tensor([f.pos_ids for f in features], dtype=torch.long)
        all_rel_ids = torch.tensor([f.rel_ids for f in features], dtype=torch.long)
        # logger.warning(all_rel_ids.shape)
        # all_rel_ids=torch.cat()
        if features[0].know_ids :
            all_know_ids = torch.tensor([f.know_ids for f in features], dtype=torch.long)
            # all_know_adj=torch.tensor([f.know_adj_mar for f in features], dtype=torch.long)
        else:
            all_know_ids = torch.tensor([0 for f in features], dtype=torch.long)
            # all_know_adj = torch.tensor([0 for f in features], dtype=torch.long)



        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_boundary_ids, all_pos_ids,
                                all_rel_ids, all_know_ids)
        return dataset
