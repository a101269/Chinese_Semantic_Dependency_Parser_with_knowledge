#    Author:  a101269
#    Date  :  2020/4/23
from constants import *

class InputFeatures(object):
    def __init__(self, text, label, input_id, label_id, input_mask):
        self.text = text
        self.label = label
        self.input_id = input_id
        self.label_id = label_id
        self.input_mask = input_mask


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = {}
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def load_file(file_path):
    contents = open(file_path, encoding='utf-8').readlines()
    text = []
    label = []
    texts = []
    labels = []
    for line in contents:
        if line != '\n':
            line = line.strip().split('\t')
            text.append(line[0])
            label.append(line[-1])
        else:
            texts.append(text)
            labels.append(label)
            text = []
            label = []
    return texts, labels


def load_data(file_path, max_length, label_dic, vocab):
    texts, labels = load_file(file_path)
    assert len(texts) == len(labels)
    result = []
    for i in range(len(texts)):
        assert len(texts[i]) == len(labels[i])
        token = texts[i]
        label = labels[i]
        if len(token) > max_length - 2:
            token = token[0:(max_length - 2)]
            label = label[0:(max_length - 2)]
        tokens_f = ['[CLS]'] + token + ['[SEP]']
        label_f = ["<start>"] + label + ['<eos>']
        input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in tokens_f]
        label_ids = [label_dic[i] for i in label_f]
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_length:
            input_ids.append(0)
            input_mask.append(0)
            label_ids.append(label_dic['<pad>'])
        assert len(input_ids) == max_length
        assert len(input_mask) == max_length
        assert len(label_ids) == max_length
        feature = InputFeatures(text=tokens_f, label=label_f, input_id=input_ids, input_mask=input_mask,
                                label_id=label_ids)
        result.append(feature)
    return result


def recover_label(pred_var, gold_var, l2i_dic, i2l_dic):
    assert len(pred_var) == len(gold_var)
    pred_variable = []
    gold_variable = []
    for i in range(len(gold_var)):
        start_index = gold_var[i].index(l2i_dic['<start>'])  # 0:"O",1:u'B-SUB'，，定位起始位置
        end_index = gold_var[i].index(l2i_dic['<eos>'])
        pred_variable.append(pred_var[i][start_index:end_index])
        gold_variable.append(gold_var[i][start_index:end_index])

    pred_label = []
    gold_label = []
    for j in range(len(gold_variable)):
        pred_label.append([i2l_dic[t] for t in pred_variable[j]])
        gold_label.append([i2l_dic[t] for t in gold_variable[j]])
    fw = open('pred.txt', 'w')
    for ps in pred_label:
        for p in ps:
            fw.write(p + '\t')
        fw.write('\n')
    return pred_label, gold_label


def get_ner_fmeasure(golden_lists, predict_lists, label_type="BMES"):
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_full = []
    counts = {'SUB': {'right': 0, 'gold': 0, 'pred': 0}, 'OBJ': {'right': 0, 'gold': 0, 'pred': 0},
              'WOR': {'right': 0, 'gold': 0, 'pred': 0},
              'PRO': {'right': 0, 'gold': 0, 'pred': 0}, 'RES': {'right': 0, 'gold': 0, 'pred': 0}}
    measures = {'SUB': {'precision': 0.0, 'recall': 0.0, 'f': 0.0}, 'OBJ': {'precision': 0.0, 'recall': 0.0, 'f': 0.0},
                'WOR': {'precision': 0.0, 'recall': 0.0, 'f': 0.0},
                'PRO': {'precision': 0.0, 'recall': 0.0, 'f': 0.0}, 'RES': {'precision': 0.0, 'recall': 0.0, 'f': 0.0}}
    right_tag = 0
    all_tag = 0
    for idx in range(0, sent_num):
        # word_list = sentence_lists[idx]
        gold_factors = {'SUB': [], 'OBJ': [], 'PRO': [], 'WOR': [], 'RES': []}
        pred_factors = {'SUB': [], 'OBJ': [], 'PRO': [], 'WOR': [], 'RES': []}

        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)
        if label_type == "BMES":
            gold_matrix = get_ner_BMES(golden_list)
            pred_matrix = get_ner_BMES(predict_list)
            gold_items = []
            pred_items = []
            beg_end_tag(gold_matrix, gold_items)
            beg_end_tag(pred_matrix, pred_items)
            for item in gold_items:
                gold_factors[item[2]].append(item)
            for item in pred_items:
                pred_factors[item[2]].append(item)
        else:
            gold_matrix = get_ner_BIO(golden_list)
            pred_matrix = get_ner_BIO(predict_list)
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))  # intersection()返回两个或更多集合的交集。
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner

        for k in gold_factors.keys():
            counts[k]['right'] += len(list(set(gold_factors[k]).intersection(set(pred_factors[k]))))
            counts[k]['gold'] += len(gold_factors[k])
            counts[k]['pred'] += len(pred_factors[k])

    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)
    if predict_num == 0:
        precision = -1
    else:
        precision = (right_num + 0.0) / predict_num
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num + 0.0) / golden_num
    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)
    accuracy = (right_tag + 0.0) / all_tag

    # print "Accuracy: ", right_tag,"/",all_tag,"=",accuracy
    factors = ['SUB', 'OBJ', 'PRO', 'WOR', 'RES']
    for k in factors:
        if counts[k]['pred'] == 0:
            measures[k]['precision'] = -1
        else:
            measures[k]['precision'] = (counts[k]['right'] + 0.0) / counts[k]['pred']
        if counts[k]['gold'] == 0:
            measures[k]['recall'] = -1
        else:
            measures[k]['recall'] = (counts[k]['right'] + 0.0) / counts[k]['gold']
        if (measures[k]['precision'] == -1) or (measures[k]['recall'] == -1) or (
                measures[k]['precision'] + measures[k]['recall']) <= 0.:
            measures[k]['f'] = -1
        else:
            measures[k]['f'] = 2 * measures[k]['precision'] * measures[k]['recall'] / (
                        measures[k]['precision'] + measures[k]['recall'])
    print("gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num)
    return accuracy, precision, recall, f_measure, measures


def beg_end_tag(info_list, res):
    for info in info_list:
        position = info.split(']')[0].strip('[')
        position = position.split(',')
        if len((position)) == 2:
            begin = position[0]
            end = position[1]
        else:
            begin = position[0]
            end = position[0]
        label = info.split(']')[1]
        item = (begin, end, label)
        res.append(item)


def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string


def get_ner_BMES(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag != '':  # 上一个字符已被处理
                tag_list.append(whole_tag + ',' + str(i - 1))  # 上一个字符的
            whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)  # 字符串中old替换成new替换不超过 max次
            index_tag = current_label.replace(begin_label, "", 1)  # 只是将B-去掉，后面的类型还存在

        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = current_label.replace(single_label, "", 1) + '[' + str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""  # 避免被添加两次
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i))  # whole 理论上应与begin的标签一致，如此得：标签【开始位置，结束位置
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])  # i标签
            stand_matrix.append(insert_list)
    # print(stand_matrix)
    return stand_matrix

def get_ner_BIO(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
                index_tag = current_label.replace(begin_label,"",1)
            else:
                tag_list.append(whole_tag + ',' + str(i-1))
                whole_tag = current_label.replace(begin_label,"",1)  + '[' + str(i)
                index_tag = current_label.replace(begin_label,"",1)

        elif inside_label in current_label:
            if current_label.replace(inside_label,"",1) == index_tag:
                whole_tag = whole_tag
            else:
                if (whole_tag != '')&(index_tag != ''):
                    tag_list.append(whole_tag +',' + str(i-1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '')&(index_tag != ''):
                tag_list.append(whole_tag +',' + str(i-1))
            whole_tag = ''
            index_tag = ''

    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix

def save_model(path, model, epoch):
    pass


def load_model(path, model):
    return model


if __name__ == "__main__":
    pass




