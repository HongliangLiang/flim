# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open
from sklearn.metrics import f1_score
from scipy import sparse
import json
csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None,feature=None):
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
        self.text_b = text_b
        self.label = label
        self.feature=feature


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id,feature):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.feature=feature

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            lines = []
            for line in f.readlines():
                line = line.strip().split('<CODESPLIT>')
                if len(line) != 5:
                    continue
                lines.append(line)
            return lines


class CodesearchProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""
    # def get_train_examples(self, data_dir, train_file,prefix):
    #     """See base class."""
    #     logger.info("LOOKING AT {}".format(os.path.join(data_dir, train_file)))
    #     return self._create_examples_add_feature(
    #         self._read_tsv(os.path.join(data_dir, train_file)), "train",prefix)
    #
    # def get_dev_examples(self, data_dir, dev_file,prefix):
    #     """See base class."""
    #     print('data_dir',data_dir)
    #     print('dev_file',dev_file)
    #     print('prefix',prefix)
    #     logger.info("LOOKING AT {}".format(os.path.join(data_dir, dev_file)))
    #     return self._create_examples_add_feature(
    #         self._read_tsv(os.path.join(data_dir, dev_file)), "dev",prefix)
    #
    # def get_test_examples(self, data_dir, test_file,prefix):
    #     """See base class."""
    #     logger.info("LOOKING AT {}".format(os.path.join(data_dir, test_file)))
    #     return self._create_examples_add_feature(
    #         self._read_tsv(os.path.join(data_dir, test_file)), "test",prefix)
    def get_train_examples(self, data_dir, train_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, train_file)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, train_file)), "train")

    def get_dev_examples(self, data_dir, dev_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, dev_file)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, dev_file)), "dev")

    def get_test_examples(self, data_dir, test_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, test_file)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, test_file)), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            # if (set_type == 'test'):#TODO 这个当以test进来时，label全是1 所以输出的验证标准就没有用了
            #     label = self.get_labels()[0]
            # else:
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        if (set_type == 'test'):
            return examples, lines
        else:
            return examples

    # 测试增加手动特征
    def load_features(self,file_prefix, commit):
        file_path = file_prefix + '_' + commit[0:7] + '_features.npz'
        features_data = sparse.load_npz(file_path).tocsr()
        return features_data

    def load_filenames(self,file_prefix, commit):
        file_path = file_prefix + '_' + commit[0:7] + '_files'
        with open(file_path, 'r') as f:
            files_list = json.load(f)
            return files_list
    def _create_examples_add_feature(self, lines, set_type,file_prefix):
        """Creates examples for the training and dev sets.
            file_prefix:'/data/hdj/tracking_buggy_files/swt/swt'
        """
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            # if (set_type == 'test'):#TODO 这个当以test进来时，label全是1 所以输出的验证标准就没有用了
            #     label = self.get_labels()[0]
            # else:
            label = line[0]
            idx=line[1]
            commit=idx.split('_')[0]
            filename=idx.split('_')[1]
            files_list = self.load_filenames(file_prefix, commit)
            features_data = self.load_features(file_prefix, commit)
            feature=features_data[files_list.index(filename)].toarray()[0][:-1]
            # print('feature shape',len(feature),type(feature))
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,feature=feature))
        if (set_type == 'test'):
            return examples, lines
        else:
            return examples

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)#[:50]

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            # logger.info("tokens: %s" % " ".join(
            #     [str(x) for x in tokens]))
            # logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            # logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            # logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          feature=example.feature))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

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


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "codesearch":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)

def calculate_same_value(labels_sorted, test_p_sorted, start_pos):
    i = start_pos
    num_same = 0
    num_p = 0
    while test_p_sorted[start_pos] == test_p_sorted[i]:
        num_same = num_same + 1
        if labels_sorted[i] ==1 : num_p = num_p + 1
        i = i + 1
        if i == len(labels_sorted ): break
    return num_p, num_same
def eval_mrr(test_p, labels):#在第二维相似度得分，真实标签
    test_p_sorted = test_p
    test_p_index = sorted(range(len(test_p_sorted)), key=lambda k: test_p_sorted[k], reverse=True)  # 降序排序
    test_p_sorted = sorted(test_p, reverse=True)

    labels_sorted = []
    for index in test_p_index:
        labels_sorted.append(labels[index])

    top_num = 10
    top10rank = 0
    for i in range(top_num):
        if (labels_sorted[i] == 1):
            '''
            num_p, num_s = calculate_same_value(labels_sorted, test_p_sorted, i)
            num_r = top_num - i
            if num_p > (num_s-num_r):
                top10rank = 1
                break
            v1 = perm(num_s-num_r, num_p)*perm(num_s-num_p,num_s-num_p)
            v2 = perm(num_s,num_s)
            top10rank = 1-(float)((float)(v1)/(float)(v2))
            if top10rank > 1: top10rank=1
            if top10rank!=top10rank: top10rank=1
            break

    return top10rank

    '''
            top10rank = 1
            break
    num_p, num_s = calculate_same_value(labels_sorted, test_p_sorted, 10)
    if (num_p >= 1): top10rank = 1  # 统计在第十位并列排名相同的文件中，是否含有相关文件

    MRRrank = 0.0
    for i in range(len(labels_sorted)):
        if (labels_sorted[i] == 1):
            MRRrank = MRRrank + float(1 / (i + 1))

    MAPrank = 0.0
    pos_num = 0
    for i in range(len(labels_sorted)):
        if (labels_sorted[i] == 1):
            pos_num = pos_num + 1
            MAPrank = MAPrank + float(pos_num / (i + 1))

    MAPrank = float(MAPrank / pos_num)
    MRRrank = float(MRRrank / pos_num)

    return top10rank, MRRrank, MAPrank

processors = {
    "codesearch": CodesearchProcessor,
}

output_modes = {
    "codesearch": "classification",
}

GLUE_TASKS_NUM_LABELS = {
    "codesearch": 2,
}
