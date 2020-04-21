#    Author:  a101269
#    Date  :  2020/3/5

"""
A wrapper/loader for the official conll-u format files.
"""
import os
import io

FIELD_NUM = 10

FIELD_TO_IDX = {'id': 0, 'word': 1, 'lemma': 2, 'upos': 3, 'xpos': 4, 'feats': 5, 'head': 6, 'deprel': 7, 'deps': 8,
                'misc': 9}


class CoNLLFile():
    def __init__(self, filename=None, input_str=None, ignore_gapping=True):
        # If ignore_gapping is True, all words that are gap fillers (identified with a period in
        # the sentence index) will be ignored.  如果为真，则将忽略所有空白填充词(在句子索引中使用句点标识)。

        self.ignore_gapping = ignore_gapping  # gapping 不紧密接触;
        if filename is not None and not os.path.exists(filename):
            raise Exception("File not found at: " + filename)
        if filename is None:
            assert input_str is not None and len(input_str) > 0
            self._file = input_str
            self._from_str = True
        else:
            self._file = filename
            self._from_str = False

    def load_all(self):
        """ Trigger all lazy initializations so that the file is loaded."""
        _ = self.sents
        _ = self.num_words

    def load_conll(self):
        """
        Load data into a list of sentences, where each sentence is a list of lines,
        and each line is a list of conllu fields.
        """
        sents, cache = [], []
        if self._from_str:
            infile = io.StringIO(self.file)
        else:
            infile = open(self.file, encoding='utf-8')
        while True:
            line = infile.readline()
            if len(line) == 0:  # 最后一行，"\n" 长为1
                break
            line = line.strip()
            if len(line) == 0:  # "\n" 被去掉
                if len(cache) > 0:
                    sents.append(cache)  # cache 为词信息列表的列表，so sents为 列表的列表的列表
                    cache = []
            else:
                if line.startswith('#'):  # skip comment line  ？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
                    continue
                array = line.split('\t')
                if self.ignore_gapping and '.' in array[0]:  # 第一列为序号，为什么有 . ？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
                    continue
                if len(array) !=10:
                    print('====================')
                    print(len(array))
                    print(array)
                assert len(array) == FIELD_NUM  # 列数超过10，与标准不同
                cache += [array]  # cache 为词信息列表的列表
        if len(cache) > 0:  # 最后一句
            sents.append(cache)
        if not self._from_str:
            infile.close()
        return sents  # 列表的列表的列表

    @property
    def file(self):
        return self._file

    @property
    def sents(self):  # 有必要非弄个这东西吗 ？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
        if not hasattr(self, '_sents'):
            self._sents = self.load_conll()  # 为什么需要判断？？？？？？？？？？？？？
        return self._sents  # 列表的列表的列表

    def __len__(self):
        return len(self.sents)

    @property
    def num_words(self):
        """ Num of total words, after multi-word expansion."""
        if not hasattr(self, '_num_words'):
            n = 0
            for sent in self.sents:
                for ln in sent:
                    if '-' not in ln[0]:
                        n += 1
            self._num_words = n
        return self._num_words

    def get(self, fields,
            as_sentences=False):  # data=conll_file.get(['word', 'upos', 'head', 'deprel'], as_sentences=True)
        """ Get fields from a list of field names. If only one field name is provided, return a list
        of that field; if more than one, return a list of list. Note that all returned fields are after
        multi-word expansion.   load_conll()与 get() 完全可以合并在一起 self.sents调用load_conll()
        """
        assert isinstance(fields, list), "Must provide field names as a list."
        assert len(fields) >= 1, "Must have at least one field."
        field_idxs = [FIELD_TO_IDX[f.lower()] for f in fields]
        """ FIELD_TO_IDX = {'id': 0, 'word': 1, 'lemma': 2, 'upos': 3, 'xpos': 4, 'feats': 5, 'head': 6, 'deprel': 7, 'deps': 8, 'misc': 9}
        """
        results = []
        for sent in self.sents:  # self.sents 列表的列表的列表     # load_conll()与 get() 完全可以合并在一起
            cursent = []  # sent就是 词各列组成列表的列表[[],[]],各个词组成一句话
            for ln in sent:
                if '-' in ln[0]:  # skip
                    continue
                if len(field_idxs) == 1:  # 什么情况会出现只需要一列信息？？？？？？？？？？？？？？？？？？？？？？？
                    cursent += [ln[field_idxs[0]]]
                else:
                    cursent += [
                        [ln[fid] for fid in field_idxs]]  # cursent为保存 id word pos head deprel 的列表的列表，其余列舍去，相当于简化的sent

            if as_sentences:
                results.append(cursent)
            else:
                results += cursent  # 各句子的词不区分，全在一个列表里，为什么有这种操作？？？？？？？？？？？？？？？?
        return results

    def set(self, fields, contents):
        """ Set fields based on contents. If only one field (singleton list) is provided, then a list of content will be expected; otherwise a list of list of contents will be expected.
        """
        assert isinstance(fields, list), "Must provide field names as a list."
        assert isinstance(contents, list), "Must provide contents as a list (one item per line)."
        assert len(fields) >= 1, "Must have at least one field."
        assert self.num_words == len(contents), "Contents must have the same number as the original file."
        field_idxs = [FIELD_TO_IDX[f.lower()] for f in fields]
        cidx = 0
        for sent in self.sents:
            for ln in sent:
                if '-' in ln[0]:
                    continue
                if len(field_idxs) == 1:
                    ln[field_idxs[0]] = contents[cidx]
                else:
                    for fid, ct in zip(field_idxs, contents[cidx]):
                        ln[fid] = ct
                cidx += 1
        return

    def write_conll(self, filename):
        """ Write current conll contents to file.
        """
        conll_string = self.conll_as_string()
        with open(filename, 'w', encoding='utf-8') as outfile:
            outfile.write(conll_string)
        return

    def conll_as_string(self):
        """ Return current conll contents as string
        """
        return_string = ""
        for sent in self.sents:
            for ln in sent:
                return_string += ("\t".join(ln) + "\n")
            return_string += "\n"
        return return_string

    def write_conll_with_lemmas(self, lemmas, filename):
        """ Write a new conll file, but use the new lemmas to replace the old ones."""
        assert self.num_words == len(lemmas), "Num of lemmas does not match the number in original data file."
        lemma_idx = FIELD_TO_IDX['lemma']
        idx = 0
        with open(filename, 'w') as outfile:
            for sent in self.sents:
                for ln in sent:
                    if '-' not in ln[0]:  # do not process if it is a mwt line
                        lm = lemmas[idx]
                        if len(lm) == 0:
                            lm = '_'
                        ln[lemma_idx] = lm
                        idx += 1
                    print("\t".join(ln), file=outfile)
                print("", file=outfile)
        return

    def get_mwt_expansions(self):
        word_idx = FIELD_TO_IDX['word']
        expansions = []
        src = ''
        dst = []
        for sent in self.sents:
            mwt_begin = 0
            mwt_end = -1
            for ln in sent:
                if '.' in ln[0]:
                    # skip ellipsis
                    continue

                if '-' in ln[0]:
                    mwt_begin, mwt_end = [int(x) for x in ln[0].split('-')]
                    src = ln[word_idx]
                    continue

                if mwt_begin <= int(ln[0]) < mwt_end:
                    dst += [ln[word_idx]]
                elif int(ln[0]) == mwt_end:
                    dst += [ln[word_idx]]
                    expansions += [[src, ' '.join(dst)]]
                    src = ''
                    dst = []

        return expansions

    def get_mwt_expansion_cands(self):
        word_idx = FIELD_TO_IDX['word']
        cands = []
        for sent in self.sents:
            for ln in sent:
                if "MWT=Yes" in ln[-1]:
                    cands += [ln[word_idx]]

        return cands

    def write_conll_with_mwt_expansions(self, expansions, output_file):
        """ Expands MWTs predicted by the tokenizer and write to file. This method replaces the head column with a right branching tree. """
        idx = 0
        count = 0

        for sent in self.sents:
            for ln in sent:
                idx += 1
                if "MWT=Yes" not in ln[-1]:
                    print("{}\t{}".format(idx, "\t".join(ln[1:6] + [str(idx - 1)] + ln[7:])), file=output_file)
                else:
                    # print MWT expansion
                    expanded = [x for x in expansions[count].split(' ') if len(x) > 0]
                    count += 1
                    endidx = idx + len(expanded) - 1

                    print("{}-{}\t{}".format(idx, endidx,
                                             "\t".join(['_' if i == 5 or i == 8 else x for i, x in enumerate(ln[1:])])),
                          file=output_file)
                    for e_i, e_word in enumerate(expanded):
                        print("{}\t{}\t{}".format(idx + e_i, e_word,
                                                  "\t".join(['_'] * 4 + [str(idx + e_i - 1)] + ['_'] * 3)),
                              file=output_file)
                    idx = endidx

            print("", file=output_file)
            idx = 0

        assert count == len(expansions), "{} {} {}".format(count, len(expansions), expansions)
        return


if __name__ == '__main__':
    filename = '..\dataset\sdp_text_dev.conllu'
    system_predict_file = '../Eval/sdp_text_dev_predict.conllu'
    conll = CoNLLFile(filename)
    res = conll.get(['deps'])
    print(res)
    print(conll.num_words)
    conll.set(['deps'], res)
    conll.write_conll(system_predict_file)



