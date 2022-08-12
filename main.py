# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import glob
import json
import os
import sys
import time

from tensorflow.python.estimator.exporter import BestExporter

from towers import modeling
from modeling2 import BertConfig, get_assignment_map_from_checkpoint, BertModel, create_initializer,layers_attention
from utils.tools import tokenization, optimization
import tensorflow as tf

from modeling2 import get_shape_list
from utils.tools.exporter import BestCheckpointExporter
from tensorflow.python.client import device_lib
#print("===================",device_lib.list_local_devices())
flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the data will be written.")

flags.DEFINE_string(
    "model_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "query_max_seq_length", 32,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "desc_max_seq_length", 256,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

tf.flags.DEFINE_bool("do_write", False, "Whether to write record")

tf.flags.DEFINE_bool("do_pb", False, "Whether to convert to pb")

tf.flags.DEFINE_bool("do_check", False, "Whether to do check")


flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

tf.flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_integer(
    "start_delay_secs", 1800,
    "eval_spec start delay secs"
)

flags.DEFINE_integer(
    "throttle_secs", 900,
    "eval_spec throttle secs"
)

flags.DEFINE_string(
    "metric_type", "auc",
    "exporter metric type"
)

flags.DEFINE_float(
    "threshold", 0.5,
    "eval threshold"
)

flags.DEFINE_integer(
    "sent_dim", 128,
    "sentence embedding dimension"
)

flags.DEFINE_float(
    "eps", 1.0,
    "loss eps"
)

flags.DEFINE_string(
    "dist_type", "l2",
    "vector distance type"
)


flags.DEFINE_bool(
    "pooling_mode_cls_token", True,
    "pooling_mode_cls_token"
)

flags.DEFINE_bool(
    "pooling_mode_max_tokens", False,
    "pooling_mode_max_tokens"
)

flags.DEFINE_bool(
    "pooling_mode_mean_tokens", False,
    "pooling_mode_mean_tokens"
)

flags.DEFINE_bool(
    "pooling_mode_mean_sqrt_len_tokens", False,
    "pooling_mode_mean_sqrt_len_tokens"
)


flags.DEFINE_bool(
    "feature_pv", False,
    "feature_pv"
)

flags.DEFINE_bool(
    "feature_linkcount", False,
    "feature_linkcount"
)

flags.DEFINE_bool(
    "feature_linkcount_p", False,
    "feature_linkcount_p"
)


flags.DEFINE_bool(
    "feature_coherence", False,
    "feature_coherence"
)
flags.DEFINE_bool(
    "feature_topic", False,
    "feature_topic"
)
flags.DEFINE_bool(
    "feature_offset",False,
    "feature_offset"
)
flags.DEFINE_bool(
    "feature_end",False,  
    "feature_end"
)
flags.DEFINE_float(
    "score_threshold", 0.0,
    "score_threshold"
)

flags.DEFINE_bool(
    "is_proj", False,
    "is_proj"
)

flags.DEFINE_bool(
    "eval_reverse", True,
    "eval reverse"
)

from collections import namedtuple

THRES = "threshold"
ACC = "acc"
F1 = "f1"
PREC = "precision"
RECALL = "recall"
AUC = "auc"


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid:str, query:str, mention:str, desc:str, props:list, offset:int,end:int,pv:int,coherence:int,topic:int,linkcount:int, linkcount_p:float, label:float):
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
        self.query = query
        self.mention = mention
        self.desc = desc
        self.offset=offset
        self.end =end
        self.pv = pv
        self.coherence = coherence
        self.topic = topic
        self.linkcount = linkcount
        self.linkcount_p = linkcount_p
        self.label = label
        self.props = props


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 query_input_ids,
                 query_seg_len,
                 desc_input_ids,
                 desc_seg_len,
                 desc_position_ids,
                 offset,
                 end,
                 pv,
                 coherence,
                 topic,
                 linkcount,
                 linkcount_p,
                 is_real_example=True,
                 label=0.0):
        self.query_input_ids = query_input_ids
        self.query_seg_len = query_seg_len

        self.desc_input_ids = desc_input_ids
        self.desc_seg_len = desc_seg_len
        self.desc_position_ids = desc_position_ids
        self.offset=offset
        self.end=end
        self.pv = pv
        self.coherence=coherence
        self.topic = topic
        self.linkcount = linkcount
        self.linkcount_p = linkcount_p
        self.is_real_example = is_real_example
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        lines = []
        if not os.path.exists(input_file): return lines
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            for line in reader:
                lines.append(line)
        return lines

    @classmethod
    def _read_file(cls, input_file):
        lines = []
        if not os.path.exists(input_file): return lines
        with tf.gfile.Open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                lines.append(line)
        return lines

    @classmethod
    def _read_json(cls, input_file):
        lines = []
        if not os.path.exists(input_file): return lines
        with tf.gfile.Open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                lines.append(json.loads(line))
                if len(lines) % 5000 == 0: tf.logging.info("Parse {} lines".format(len(lines)))
        return lines

CTRL_A='\x01'
CTRL_B='\x02'
CTRL_C='\x03'


class EntityLinkingProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        pass

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            idx = line['id']
            guid = "%s-%d-%s" % (set_type, i, idx)
            query = line['query']
            mention = line['mention']
            offset = int(line['offset'])
            desc = line['desc']
            #offset = query.index(mention)
            end = offset+len(mention)-1
            pv = int(line['pv'])
            coherence = int(line["coherence_v2"])
            coherence_v1 = line["coherence_v1"]
            topic = int(line["type_coo"])
            #coherence=1#先随机初始化
            #topic=1
            #linkcount = line['linkcount']
            linkcount = 0
            linkcount_p = float(line['linkcount_p'])
            label =int(float(line['label']))
            props = line['properties']
            examples.append(InputExample(
                guid=guid,
                query=query,
                mention=mention,
                desc=desc,
                props=props,
                offset=offset,
                end=end,
                pv=pv,
                coherence=coherence,
                topic=topic,
                linkcount=linkcount,
                linkcount_p=linkcount_p,
                label=label
            ))
        return examples


def convert_single_example(ex_index, example, label_list, query_max_seq_length, desc_max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            query_input_ids=[0] * query_max_seq_length,
            query_seg_len=query_max_seq_length,
            desc_input_ids=[0] * desc_max_seq_length,
            desc_position_ids=[0] * desc_max_seq_length,
            desc_seg_len=query_max_seq_length,
            offset=0,
            end=0,
            pv=0,
            coherence=0,
            topic=0,
            linkcount=0,
            linkcount_p=0.0,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    query = example.query
    offset = example.offset
    mention = example.mention

    prefix = query[0:offset]
    suffix = query[offset + len(mention): ]
    prefix_toks = tokenizer.tokenize(prefix)
    mention_toks = tokenizer.tokenize(mention)
    suffix_toks = tokenizer.tokenize(suffix)
    _truncate_seq_pair(prefix_toks, suffix_toks, max_length=query_max_seq_length - 4 - len(mention_toks))
    #mention用[BAR]标记
    query_toks = ['[CLS]'] + prefix_toks + ['[BRA]'] + mention_toks + ['[BRA]'] + suffix_toks + ['[SEP]']
    desc_toks = tokenizer.tokenize(example.desc)
    if len(desc_toks) > desc_max_seq_length - 2:
        desc_toks = desc_toks[0: desc_max_seq_length - 2]

    desc_toks = ['[CLS]'] + desc_toks + ['[SEP]']

    query_ids = tokenizer.convert_tokens_to_ids(query_toks)
    query_len = len(query_ids)




    while len(query_ids) < query_max_seq_length:
        query_ids.append(0)

    desc_position = [i for i in range(len(desc_toks))]

    #实体doc：desc+prop
    for prop in example.props:
        toks = tokenizer.tokenize(prop)
        if len(desc_toks) + len(toks) + 1 > desc_max_seq_length: break
        index = 1
        for tok in toks:
            desc_toks.append(tok)
            desc_position.append(index)
            index += 1
        desc_toks.append('[SEP]')
        desc_position.append(index)



    desc_ids = tokenizer.convert_tokens_to_ids(desc_toks)
    desc_len = len(desc_ids)


    while len(desc_ids) < desc_max_seq_length:
        desc_ids.append(0)
        desc_position.append(0)


    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("query_tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in query_toks]))
        tf.logging.info("query_input_ids: %s" % " ".join([str(x) for x in query_ids]))
        tf.logging.info("desc_tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in desc_toks]))
        tf.logging.info("desc_input_ids: %s" % " ".join([str(x) for x in desc_ids]))
        tf.logging.info("desc_position_ids: %s" % " ".join([str(x) for x in desc_position]))
        tf.logging.info("label: {}".format(example.label))

    feature = InputFeatures(
        query_input_ids=query_ids,
        query_seg_len=query_len,
        desc_input_ids=desc_ids,
        desc_position_ids=desc_position,
        desc_seg_len=desc_len,
        offset = example.offset,
        end=example.end,
        pv=example.pv,
        coherence = example.coherence,
        topic = example.topic,
        linkcount=example.linkcount,
        linkcount_p=example.linkcount_p,
        is_real_example=True,
        label=example.label
    )

    return feature


def file_based_convert_examples_to_features(
        examples, label_list, query_max_seq_length, desc_max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    tf.logging.info('query_max_seq_length:{}, desc_max_seq_length:{}...'.format(query_max_seq_length, desc_max_seq_length))

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         query_max_seq_length, desc_max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f
        def create_float_feature(values):
            f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["query_input_ids"] = create_int_feature(feature.query_input_ids)
        features["query_seq_len"] = create_int_feature([feature.query_seg_len])
        features["desc_input_ids"] = create_int_feature(feature.desc_input_ids)
        features["desc_seq_len"] = create_int_feature([feature.desc_seg_len])
        features["desc_position_ids"] = create_int_feature(feature.desc_position_ids)
        features["offset"] = create_int_feature([feature.offset])
        features["end"] = create_int_feature([feature.end])
        features["pv"] = create_int_feature([feature.pv])
        features["coherence"]=create_int_feature([feature.coherence])
        features["topic"]=create_int_feature([feature.topic])
        features["linkcount"] = create_int_feature([int(feature.linkcount)])
        features["linkcount_p"] = create_float_feature([feature.linkcount_p])
        #精排模型的输出分数作为label，float类型
        features["label"] = create_float_feature([feature.label])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, query_seq_length, desc_seq_length, is_training,
                                   drop_remainder, batch_size):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    tf.logging.info('query_input_ids len:{}, desc_input_ids len:{}...'.format(query_seq_length, desc_seq_length))
    name_to_features = {
        "query_input_ids": tf.FixedLenFeature([query_seq_length], tf.int64),
        "query_seq_len": tf.FixedLenFeature([], tf.int64),
        "desc_input_ids": tf.FixedLenFeature([desc_seq_length], tf.int64),
        "desc_position_ids": tf.FixedLenFeature([desc_seq_length], tf.int64),
        "desc_seq_len": tf.FixedLenFeature([], tf.int64),
        "offset": tf.FixedLenFeature([], tf.int64),
        "end": tf.FixedLenFeature([], tf.int64),
        "pv": tf.FixedLenFeature([], tf.int64),
        "coherence": tf.FixedLenFeature([], tf.int64),
        "topic": tf.FixedLenFeature([], tf.int64),
        "linkcount": tf.FixedLenFeature([], tf.int64),
        "linkcount_p": tf.FixedLenFeature([], tf.float32),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
        "label": tf.FixedLenFeature([], tf.float32)
    }



    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            tf.logging.info('***************key name:{}***************'.format(name))
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn():
        """The actual input function."""
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=256)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


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

def get_sent_embedding_v2(bert_mode:BertModel, name ='sent_embedding', is_proj=False, dim=64,offset=0,end=0,get_layers_att=False,get_mention_emb=False):
    output_vectors = []
    sequence_output = bert_mode.get_sequence_output()#[batch,seq_len,768]
    input_mask = bert_mode.get_input_mask()
    input_mask_extend = tf.expand_dims(input_mask, axis=2)
    #获取mention的span token embedding
    if get_mention_emb:
        length = end - offset + 1
        max_mention_length = tf.reduce_max(length)  # axis=0:每列最大值
        mention_index = tf.range(max_mention_length) + offset
        mention_mask = tf.cast(tf.less_equal(mention_index, end), tf.float32) 
        mention_index = tf.minimum(mention_index, end)
        batch_index = tf.tile(tf.expand_dims(tf.range(tf.shape(sequence_output)[0]), 1), [1, tf.shape(mention_index)[1]])
        mention_index = tf.stack([batch_index, tf.compat.v1.to_int32(mention_index)], axis=2)
        span_mention = tf.gather_nd(sequence_output, mention_index)
        mention_spand_embed = tf.reduce_mean(span_mention, 1)  # 取平均值作为mention向量
        output_vectors.append(mention_span_embed)
    if FLAGS.pooling_mode_cls_token:
        #[CLS]
        first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
        output_vectors.append(first_token_tensor)
    if FLAGS.pooling_mode_max_tokens:
        neg_inf_seq = -1e9 * tf.ones_like(sequence_output, dtype=tf.float32)
        hidden_size = sequence_output.shape[-1].value
        input_mask_extend_tile = tf.tile(input_mask_extend, [1, 1, hidden_size])
        final_seq_output = tf.where(tf.equal(input_mask_extend_tile, 1), sequence_output, neg_inf_seq)
        output_vectors.append(tf.reduce_max(final_seq_output, axis=1))
    if FLAGS.pooling_mode_mean_tokens or FLAGS.pooling_mode_mean_sqrt_len_tokens:
        sum_embedding = tf.reduce_sum(sequence_output * tf.cast(input_mask_extend, dtype=tf.float32), axis=1)
        sum_len = tf.cast(tf.reduce_sum(input_mask, axis=1, keepdims=True), dtype=tf.float32)
        if FLAGS.pooling_mode_mean_tokens: output_vectors.append(sum_embedding / sum_len)
        if FLAGS.pooling_mode_mean_sqrt_len_tokens: output_vectors.append(sum_embedding / tf.sqrt(sum_len))
    
    '''层级注意力结果'''
    if get_layers_att:
        first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
        #num_class_list=[392, 76, 116, 329, 835, 1138, 1758, 2258, 2417, 2133, 1556, 1149, 1085, 589, 387, 161, 71, 131, 8, 1]    #根据wiki的数据类别构建的层级类别
        num_class_list=[410, 81, 118, 337, 851, 1167, 1800, 2345, 2486, 2204, 1602, 1195, 1140, 615, 393, 168, 72, 135, 10, 1]    #0729新类别层级
        attention_unit_size = 512
        fc_hidden_size = 768 
        layer_output = layers_attention(sequence_output,first_token_tensor,num_class_list,attention_unit_size, fc_hidden_size)
        vector = tf.concat(layer_output, axis=1)
        hidden_size = vector.shape[-1].value
        with tf.variable_scope('layer_sent_proj'):
            output_weights = tf.get_variable("output_weights", [dim, hidden_size],initializer=tf.truncated_normal_initializer(stddev=0.02))
            layer_out = tf.matmul(vector, output_weights, transpose_b=True)
        output_vectors = [layer_out]

    if not is_proj:
        return tf.concat(output_vectors, axis=1, name=name)
    else:
        vector = tf.concat(output_vectors, axis=1)
        hidden_size = vector.shape[-1].value
        with tf.variable_scope('sent_embedding', reuse=tf.AUTO_REUSE):
            output_weights = tf.get_variable(
                "output_weights", [dim, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))
            # if is_training:
            #     # I.e., 0.1 dropout
            #     input_tensor = tf.nn.dropout(input_tensor, keep_prob=0.9)
            return tf.matmul(vector, output_weights, transpose_b=True)


def get_sent_embedding_v1(input_tensor, dim, is_training=True):
    hidden_size = input_tensor.shape[-1].value
    with tf.variable_scope('sent_embedding', reuse=tf.AUTO_REUSE):
        output_weights = tf.get_variable(
            "output_weights", [dim, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        # if is_training:
        #     # I.e., 0.1 dropout
        #     input_tensor = tf.nn.dropout(input_tensor, keep_prob=0.9)
        return tf.matmul(input_tensor, output_weights, transpose_b=True)

def get_cos_distance(vec_a, vec_b):
    modeling.assert_rank(vec_a, expected_rank=2)
    modeling.assert_rank(vec_b, expected_rank=2)
    vec_a = tf.nn.l2_normalize(vec_a, axis=1)
    vec_b = tf.nn.l2_normalize(vec_b, axis=1)
    return tf.reduce_sum(vec_a * vec_b, axis=1)

def get_l2_distance(vec_a, vec_b, is_normal=False):
    modeling.assert_rank(vec_a, expected_rank=2)
    modeling.assert_rank(vec_b, expected_rank=2)
    if is_normal:
        tf.logging.info('output vector l2 nomalize...')
        vec_a = tf.nn.l2_normalize(vec_a, axis=1)
        vec_b = tf.nn.l2_normalize(vec_b, axis=1)
    d2 = tf.reduce_sum(tf.square(vec_a - vec_b), axis=1)
    d = tf.sqrt(d2)
    return d2, d

def create_encode_model(bert_config, input_ids, seq_len, is_proj=False, dim=64):
    input_shape = get_shape_list(input_ids, expected_rank=2)
    batch_size = input_shape[0]
    seq_max_len = tf.reduce_max(seq_len)
    input_ids = tf.slice(input_ids, [0, 0], [batch_size, seq_max_len])
    input_mask = tf.sequence_mask(seq_len, maxlen=seq_max_len, dtype=tf.int32)
    segment_ids = tf.zeros_like(input_ids)

    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False)

    return get_sent_embedding_v2(model, "sent_embedding/MatMul", is_proj=FLAGS.is_proj, dim=FLAGS.sent_dim)


def create_model(bert_config, is_training,
                 a_input_ids, a_seq_len,
                 p_input_ids, p_seq_len, p_position_ids,
                 use_one_hot_embeddings, num_labels,
                 is_sharing=True,
                 link_count = None,
                 link_count_p = None,
                 coherence = None,
                 topic = None,
                 pv = None,
                 offset=None,
                 end=None):
    #其他特征
    additional_features = []
    if link_count is not None:
        link_count = tf.expand_dims(link_count, axis=1)
        link_count = link_count + 1
        link_count = tf.log(tf.cast(link_count, dtype=tf.float32))
        additional_features.append(link_count)

    if link_count_p is not None:
        link_count_p = tf.expand_dims(link_count_p, axis=1)
        additional_features.append(link_count_p)
    if coherence is not None:
        coherence = tf.expand_dims(coherence, axis=1)
        coherence = coherence+1
        coherence = tf.log(tf.cast(coherence, dtype=tf.float32))
        additional_features.append(coherence)
    if topic is not None:
        topic = tf.expand_dims(topic, axis=1)
        topic = topic + 1
        topic = tf.log(tf.cast(topic, dtype=tf.float32))
        additional_features.append(topic)
    if pv is not None:
        pv = tf.expand_dims(pv, axis=1)
        pv = pv + 1
        pv = tf.log(tf.cast(pv, dtype=tf.float32))
        additional_features.append(pv)

    input_shape = get_shape_list(a_input_ids, expected_rank=2)
    batch_size = input_shape[0]
    a_seq_max_len = tf.reduce_max(a_seq_len)
    a_input_ids = tf.slice(a_input_ids, [0, 0], [batch_size, a_seq_max_len])
    a_input_mask = tf.sequence_mask(a_seq_len, maxlen=a_seq_max_len, dtype=tf.int32)
    a_segment_ids = tf.zeros_like(a_input_ids)

    p_seq_max_len = tf.reduce_max(p_seq_len)
    p_input_ids = tf.slice(p_input_ids, [0, 0], [batch_size, p_seq_max_len])
    p_position_ids = tf.slice(p_position_ids, [0, 0], [batch_size, p_seq_max_len])
    p_input_mask = tf.sequence_mask(p_seq_len, maxlen=p_seq_max_len, dtype=tf.int32)
    p_segment_ids = tf.zeros_like(p_input_ids)


    name_1, name_2 = ('bert', 'bert') if is_sharing else ('bert_1', 'bert_2')
    """Creates a classification model."""
    model_1 = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=a_input_ids,
        input_mask=a_input_mask,
        token_type_ids=a_segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope=name_1)

    model_2 = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=p_input_ids,
        input_mask=p_input_mask,
        input_position_ids=p_position_ids,
        token_type_ids=p_segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope=name_2)

    #bert_query
    sent_emb_1 = get_sent_embedding_v2(model_1, "sent_embedding_1", is_proj=FLAGS.is_proj, dim=FLAGS.sent_dim,offset=offset,end=end,get_mention_emb=False)
    #bert_doc+层级分类注意力
    sent_emb_2 = get_sent_embedding_v2(model_2, "sent_embedding_2", is_proj=FLAGS.is_proj, dim=768, get_layers_att=False)

    with tf.variable_scope("project"):
        #Q/D/(Q-D)*topic
        features = [sent_emb_1, sent_emb_2, tf.abs(sent_emb_1 - sent_emb_2)]
        #cosine
        query_norm = tf.sqrt(tf.reduce_sum(tf.square(sent_emb_1), axis=1))
        doc_norm = tf.sqrt(tf.reduce_sum(tf.square(sent_emb_2), axis=1))
        q_d = tf.reduce_sum(tf.multiply(sent_emb_1, sent_emb_2), axis=1)
        context = q_d / (query_norm * doc_norm)
        #ffnn
        features = tf.concat(features, axis=-1)
        features = tf.layers.dense(features,bert_config.hidden_size,activation=tf.nn.relu,kernel_initializer=create_initializer(bert_config.initializer_range))
        #if len(additional_features) > 0:features.extend(additional_features)
        #features.append(context)
        #特征融合层
        for a in additional_features:
            #a=tf.convert_to_tensor(a)
            features = tf.concat([features,a], axis=-1)
        context = tf.expand_dims(context, 1)
        output_layer = tf.concat([features,context], axis=-1)
        #print("feature dim----",output_layer.shape)
        #mlp
        hidden_layer = tf.layers.dense(
            output_layer,
            bert_config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=create_initializer(bert_config.initializer_range))
        hidden_size = hidden_layer.shape[-1].value
        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())
        if is_training: hidden_layer = tf.nn.dropout(hidden_layer, keep_prob=0.9)

        logits = tf.matmul(hidden_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        return (logits, probabilities, log_probs)


def create_model_with_loss(bert_config, is_training,
                           a_input_ids, a_seq_len,
                           p_input_ids, p_seq_len, p_position_ids,
                           labels,
                           use_one_hot_embeddings, num_labels,
                           link_count = None,
                           link_count_p = None,
                           coherence = None,
                           topic = None,
                           pv = None,
                           offset=None,
                           end=None,
                           is_sharing=True):
    features = []
    if link_count is not None:
        # link_count = tf.expand_dims(link_count, axis=1)
        link_count = link_count + 1
        link_count = tf.log(tf.cast(link_count, dtype=tf.float32))
        features.append(link_count)

    if link_count_p is not None:
        # link_count_p = tf.expand_dims(link_count_p, axis=1)
        features.append(link_count_p)
    if coherence is not None:
        coherence=coherence+1
        coherence = tf.log(tf.cast(coherence, dtype=tf.float32))
        features.append(coherence)
    if topic is not None:
        topic = topic +1
        topic = tf.log(tf.cast(topic, dtype=tf.float32))
        features.append(topic)
    if pv is not None:
        # pv = tf.expand_dims(pv, axis=1)
        pv = pv + 1
        pv = tf.log(tf.cast(pv, dtype=tf.float32))
        features.append(pv)

    logits, probabilities, log_probs = create_model(bert_config, is_training,
                                            a_input_ids, a_seq_len,
                                            p_input_ids, p_seq_len, p_position_ids,
                                            use_one_hot_embeddings, num_labels,
                                            is_sharing = is_sharing, link_count = link_count, link_count_p = link_count_p,coherence = coherence,topic = topic, pv = pv,offset=offset,end=end)
    with tf.variable_scope("loss"):
        postive = tf.expand_dims(labels, axis=1)
        negtive = tf.expand_dims(1.0 - labels, axis=1)
        labels = tf.concat([negtive, postive], axis=-1)
        per_example_loss = -tf.reduce_sum(labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, per_example_loss, logits, probabilities)







def get_record_num(_file):
    with open(_file, 'r') as rp:
        return json.load(rp)['num']


def set_record_num(_file, num):
    with open(_file, 'w') as wp:
        json.dump({'num': num}, wp, ensure_ascii=False)


def build_estimator(bert_config, label_list, num_train_steps, num_warmup_steps):
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(
        session_config=sess_config,
        model_dir=FLAGS.model_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=False,
        use_one_hot_embeddings=False)

    return tf.estimator.Estimator(model_fn=model_fn, config=run_config)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        query_input_ids = features["query_input_ids"]
        query_seq_len = features["query_seq_len"]

        desc_input_ids = features["desc_input_ids"]
        desc_seq_len = features["desc_seq_len"]
        desc_pos_ids = features["desc_position_ids"]
        label = features["label"]

        pv = features["pv"] if FLAGS.feature_pv else None
        coherence = features["coherence"] if FLAGS.feature_coherence else None
        topic = features["topic"] if FLAGS.feature_topic else None 
        linkcount = features["linkcount"] if FLAGS.feature_linkcount else None
        linkcount_p = features["linkcount_p"] if FLAGS.feature_linkcount_p else None

        offset = features["offset"]
        end = features["end"]
        is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        total_loss, per_example_loss, logits, probabilities = create_model_with_loss(bert_config, is_training,
                               query_input_ids, query_seq_len,
                               desc_input_ids, desc_seq_len, desc_pos_ids,
                               label, True, 2,
                               link_count=linkcount,
                               link_count_p=linkcount_p,
                               coherence = coherence,
                               topic = topic,
                               pv=pv,
                               offset=offset,
                               end=end,
                               is_sharing=False)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        print('-------------init_checkpoint:',init_checkpoint)
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            assignment_map_list = []
            for _, v in assignment_map.items():
                while len(v) > len(assignment_map_list): assignment_map_list.append({})
            for k, v in assignment_map.items():
                for idx, vv in enumerate(v):
                    assignment_map_list[idx][k] = vv

            if use_tpu:

                def tpu_scaffold():
                    for assignment_map_item in assignment_map_list:
                        tf.train.init_from_checkpoint(init_checkpoint, assignment_map_item)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                #print('assignment_map_list:',len(assignment_map_list))
                for assignment_map_item in assignment_map_list:
                    #print('assignment_map_item:',len(assignment_map_item))
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map_item)
        
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.logging.info("Start to train...")
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, False)

            logging_hook = tf.train.LoggingTensorHook({"step": tf.train.get_global_step(), "loss": total_loss},
                                                      every_n_iter=50)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[logging_hook])
        elif mode == tf.estimator.ModeKeys.EVAL:
            tf.logging.info("Start to eval...")

            def metric_fn(per_example_loss, labels, threshold, logits, probabilities, is_real_example):
                # threshold_ = FLAGS.threshold * tf.ones_like(probabilities, dtype=tf.float32)
                # logit_ones = tf.ones_like(logits, dtype=tf.float32)
                # logit_zeros = tf.zeros_like(logits, dtype=tf.float32)
                # logits = tf.where(probabilities > threshold_, logit_ones, logit_zeros)
                threshold = tf.ones_like(labels, dtype=tf.float32) * threshold
                label_ids = tf.cast(labels > threshold, dtype=tf.int32)
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)

                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)

                # accuracy
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)

                # precision
                precision = tf.metrics.precision(labels=label_ids, predictions=predictions, weights=is_real_example)

                # recall
                recall = tf.metrics.recall(labels=label_ids, predictions=predictions, weights=is_real_example)

                # #auc
                #
                auc = tf.metrics.auc(labels=label_ids, predictions=probabilities[:, 1:], weights=is_real_example)

                # f1

                calc_f1 = lambda p, r: p * r / (p + r) * 2
                f1 = (calc_f1(precision[0], recall[0]), calc_f1(precision[1], recall[1]))

                return {
                    "accuracy": accuracy,
                    "eval_loss": loss,
                    "eval_precision": precision,
                    "eval_recall": recall,
                    "eval_auc": auc,
                    "eval_f1": f1
                }

            eval_metrics = metric_fn(per_example_loss, label, FLAGS.score_threshold, logits, probabilities, is_real_example)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics
            )
        else:
            tf.logging.info("Start to predict...")

            predictions = {
                "probabilities": probabilities}

            predict_metric = {
                "probabilities": probabilities

            }

            export_outputs = {
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
                    predict_metric)}

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs
            )
        return output_spec

    return model_fn



def serving_input_receiver_fn(query_seq_length, desc_seq_length):
    def _input_receiver_fn():
        input_to_features = {
            "query_input_ids": tf.placeholder(dtype=tf.int32, shape=[None, query_seq_length], name="query_input_ids"),
            "query_seq_len": tf.placeholder(dtype=tf.int32, shape=[None], name="query_seq_len"),
            "desc_input_ids": tf.placeholder(dtype=tf.int32, shape=[None, desc_seq_length], name="desc_input_ids"),
            "desc_position_ids": tf.placeholder(dtype=tf.int32, shape=[None, desc_seq_length], name="desc_position_ids"),
            "desc_seq_len": tf.placeholder(dtype=tf.int32, shape=[None], name="desc_seq_len"),
            "label": tf.placeholder(dtype=tf.float32, shape=[None], name="label"),
            "is_real_example": tf.placeholder(dtype=tf.int32, shape=[None], name="is_real_example"),
        }
        if FLAGS.feature_pv: input_to_features["pv"] = tf.placeholder(dtype=tf.int32, shape=[None], name="pv")
        if FLAGS.feature_coherence: input_to_features["coherence"] = tf.placeholder(dtype=tf.int32, shape=[None], name="coherence")
        if FLAGS.feature_topic: input_to_features["topic"] = tf.placeholder(dtype=tf.int32, shape=[None], name="topic")
        if FLAGS.feature_linkcount: input_to_features["linkcount"] = tf.placeholder(dtype=tf.int32, shape=[None], name="linkcount")
        if FLAGS.feature_linkcount_p: input_to_features["linkcount_p"] = tf.placeholder(dtype=tf.float32, shape=[None],
                                                                                    name="linkcount_p")
        if FLAGS.feature_offset: input_to_features["offset"] = tf.placeholder(dtype=tf.int32, shape=[None], name="offset")
        if FLAGS.feature_end: input_to_features["end"] = tf.placeholder(dtype=tf.int32, shape=[None], name="end")
        return tf.estimator.export.build_raw_serving_input_receiver_fn(input_to_features)
    
    return _input_receiver_fn


def metric_bigger(best_eval_result, current_eval_result, default_key):
    if not best_eval_result or default_key not in best_eval_result:
        raise ValueError(
            'best_eval_result cannot be empty or no loss is found in it.')

    if not current_eval_result or default_key not in current_eval_result:
        raise ValueError(
            'current_eval_result cannot be empty or no loss is found in it.')

    return best_eval_result[default_key] < current_eval_result[default_key]


def metric_smaller(best_eval_result, current_eval_result, default_key):
    if not best_eval_result or default_key not in best_eval_result:
        raise ValueError(
            'best_eval_result cannot be empty or no loss is found in it.')

    if not current_eval_result or default_key not in current_eval_result:
        raise ValueError(
            'current_eval_result cannot be empty or no loss is found in it.')

    return best_eval_result[default_key] > current_eval_result[default_key]


def metric_accuracy_bigger(best_eval_result, current_eval_result):
    return metric_bigger(best_eval_result, current_eval_result, "eval_accuracy")


def metric_recall_bigger(best_eval_result, current_eval_result):
    return metric_bigger(best_eval_result, current_eval_result, "eval_recall")


def metric_precision_bigger(best_eval_result, current_eval_result):
    return metric_bigger(best_eval_result, current_eval_result, "eval_precision")


def metric_loss_smaller(best_eval_result, current_eval_result):
    return metric_smaller(best_eval_result, current_eval_result, "eval_loss")


def metric_f1_bigger(best_eval_result, current_eval_result):
    return metric_bigger(best_eval_result, current_eval_result, "eval_f1")


def metric_auc_bigger(best_eval_result, current_eval_result):
    return metric_bigger(best_eval_result, current_eval_result, "eval_auc")


def metric_auc_and_recall_bigger(best_eval_result, current_eval_result):
    return metric_auc_bigger(best_eval_result, current_eval_result) and metric_f1_bigger(best_eval_result,
                                                                                         current_eval_result)


metric_type = {
    'auc': metric_auc_bigger,
    'f1': metric_f1_bigger,
    'accuracy': metric_accuracy_bigger,
    'precision': metric_precision_bigger,
    'recall': metric_recall_bigger,
    'loss': metric_loss_smaller
}


def do_convert_pb():
    # tf.logging.info("Start convert to pb file...")
    # context_token_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="context_token_ids")
    # context_length = tf.placeholder(dtype=tf.int32, shape=[None], name="context_length")
    #
    # config = BertConfig.from_json_file(FLAGS.bert_config_file)
    # sent_embedding = create_encode_model(config, context_token_ids, context_length, is_proj=FLAGS.is_proj, dim=FLAGS.sent_dim)
    # sess_config = tf.ConfigProto(allow_soft_placement=True)
    # sess_config.gpu_options.allow_growth = True
    #
    # # for n in tf.get_default_graph().as_graph_def().node:
    # #     tf.logging.info("tensor name: {}, tensor info:{}".format(n.name, n))
    #
    # with tf.Session(config=sess_config) as sess:
    #     tf.logging.info('Init from checkpoint {}'.format(FLAGS.init_checkpoint))
    #     tvars = tf.trainable_variables()
    #     (assignment_map, initialized_variable_names
    #      ) = get_assignment_map_from_checkpoint(tvars, FLAGS.init_checkpoint)
    #
    #     tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)
    #     tf.logging.info("**** Trainable Variables ****")
    #     for var in tvars:
    #         init_string = ""
    #         if var.name in initialized_variable_names:
    #             init_string = ", *INIT_FROM_CKPT*"
    #         tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
    #                         init_string)
    #
    #     sess.run(tf.global_variables_initializer())
    #     sess.run(tf.local_variables_initializer())
    #
    #     inputs = {
    #         "context_token_ids": tf.saved_model.utils.build_tensor_info(context_token_ids),
    #         "context_length": tf.saved_model.utils.build_tensor_info(context_length)
    #     }
    #     outputs = {
    #         "sent_embedding/MatMul": tf.saved_model.utils.build_tensor_info(sent_embedding)
    #     }
    #
    #     signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs,
    #                                                                        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
    #
    #     signature_java = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, 'inference_sig_name')
    #     builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(FLAGS.output_dir, "pb_output"))
    #
    #     builder.add_meta_graph_and_variables(
    #         sess,
    #         [tf.saved_model.tag_constants.SERVING],
    #         {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature,
    #          "inference_signature": signature_java},
    #         main_op=tf.tables_initializer()
    #     )
    #
    #     builder.save()
    tf.logging.info("Finish convert to pb file!")


def _get_best_ckp(dir):
    files = [f for f in tf.gfile.ListDirectory(dir) if f.endswith('.index')]
    files.sort(key=lambda x: int(x[:-6].split('-')[1]), reverse=True)
    if not files: return ''
    tf.logging.info('Get files:{}'.format(','.join(files)))
    if len(files) > 1:
        remove_names = ['.'.join(file_name.split('.', 2)[0:-1]) for file_name in files[1:]]
        # tf.logging.info('remove names:{}'.format(','.join(remove_names)))
        for f in glob.glob(dir + "/*", recursive=False):
            # tf.logging.info('file path:{}'.format(f))
            for name in remove_names:
                if (name + '.') in f:
                    tf.logging.info('Start to delete {}'.format(f))
                    os.remove(f)
    return files[0]

def get_result(checkpoint, input_fn, examples, model):
    tf.logging.info('Get result from ckp:{}'.format(checkpoint))
    reader = tf.train.NewCheckpointReader(checkpoint)
    global_step = reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)
    result = model.predict(
        input_fn=input_fn, yield_single_examples=True, checkpoint_path=checkpoint)

    label_scores = []
    for (i, prediction) in enumerate(result):
        p_dist = float(prediction["p_dist"])
        n_dist = float(prediction["n_dist"])
        if i >= len(examples):
            break
        label = examples[i]
        if label == 1: label_scores.append((p_dist, label))
        else: label_scores.append((n_dist, label))

    eval_metric = find_best_thresh(label_scores, is_reverse=FLAGS.eval_reverse)
    auc = find_best_auc(label_scores, is_reverse=FLAGS.eval_reverse)
    eval_metric[AUC] = auc
    tf.logging.info("At step {}, metric type:{}, metric:{}".format(global_step, type(eval_metric), json.dumps(eval_metric, ensure_ascii=False)))
    return global_step, eval_metric


def find_best_auc(scores, is_reverse=True):
    sorted_scores = sorted(scores, key=lambda k: k[0], reverse=is_reverse)
    pos_num = len([label for _, label in scores if label == 1])
    neg_num = len(sorted_scores) - pos_num

    if pos_num == 0 or neg_num == 0: return 0.0
    else:
        neg_sum = 0
        pos_gt_neg = 0
        for p, l in sorted_scores:
            if l == 1: pos_gt_neg += neg_sum
            else: neg_sum += 1

        return float(pos_gt_neg) / (pos_num * neg_num)


def find_best_thresh(scores, is_reverse=True):
    sorted_scores = sorted(scores, key=lambda k: k[0], reverse=is_reverse)
    best_thresh = sys.float_info.max
    best_f1 = sys.float_info.min
    best_acc = sys.float_info.min
    best_precision = sys.float_info.min
    best_recall = sys.float_info.min
    tfn = 0
    ffn = 0

    tpn = 0
    fpn = 0
    for score, label in sorted_scores:
        if label == 1: tpn += 1
        else: fpn += 1

    for index, item in enumerate(sorted_scores):
        score, label = item
        if label == 1:
            tpn -= 1
            ffn += 1
        else:
            fpn -= 1
            tfn += 1
        if index == len(sorted_scores) - 1: continue
        try:
            precision = float(tpn) / float(tpn + fpn)
            recall = float(tpn) / float(tpn + ffn)
            f1 = 2 * precision * recall / (precision + recall)
            accuracy = float(tpn + tfn) / len(scores)
        except:
            tf.logging.info("Exception at:{}, tfn:{}, ffn:{}, tpn:{}, fpn:{}".format(index, tfn, ffn, tpn, fpn))
            continue
        if f1 > best_f1:
            best_f1 = f1
            best_acc = accuracy
            best_precision = precision
            best_recall = recall
            best_thresh = score

    return {
        THRES: best_thresh,
        ACC: best_acc,
        F1: best_f1,
        PREC: best_precision,
        RECALL: best_recall
            }





def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict and not FLAGS.do_write and not FLAGS.do_pb and not FLAGS.do_check:
        raise ValueError(
            "At least one of `do_train`, `do_eval`, `do_write`, `do_predict', `do_check` or 'do_pb' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.query_max_seq_length > bert_config.max_position_embeddings or FLAGS.desc_max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use query sequence length %d or desc sequence length %d, because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.query_max_seq_length, FLAGS.desc_max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    processor = EntityLinkingProcessor()

    # label_list = processor.get_labels()
    label_list = []

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case, is_original=True)

    if FLAGS.do_write:
        dev_examples = processor.get_dev_examples(FLAGS.data_dir)
        dev_file = os.path.join(FLAGS.output_dir, "dev.tf_record")
        dev_meta = os.path.join(FLAGS.output_dir, "dev_meta.json")
        set_record_num(dev_meta, len(dev_examples))
        while len(dev_examples) % FLAGS.eval_batch_size != 0:
            dev_examples.append(PaddingInputExample())
        file_based_convert_examples_to_features(
            dev_examples, label_list, FLAGS.query_max_seq_length, FLAGS.desc_max_seq_length, tokenizer, dev_file)

        import random
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        random.shuffle(train_examples)
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.query_max_seq_length, FLAGS.desc_max_seq_length, tokenizer, train_file)
        train_meta = os.path.join(FLAGS.output_dir, "train_meta.json")
        set_record_num(train_meta, len(train_examples))

        test_examples = processor.get_test_examples(FLAGS.data_dir)
        test_file = os.path.join(FLAGS.output_dir, "test.tf_record")
        test_meta = os.path.join(FLAGS.output_dir, "test_meta.json")
        set_record_num(test_meta, len(test_examples))
        while len(test_examples) % FLAGS.predict_batch_size != 0:
            test_examples.append(PaddingInputExample())
        file_based_convert_examples_to_features(
            test_examples, label_list, FLAGS.query_max_seq_length, FLAGS.desc_max_seq_length, tokenizer, test_file)

    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    dev_file = os.path.join(FLAGS.output_dir, "dev.tf_record")
    test_file = os.path.join(FLAGS.output_dir, "test.tf_record")

    train_meta = os.path.join(FLAGS.output_dir, "train_meta.json")
    num_train_examples = get_record_num(train_meta)

    dev_meta = os.path.join(FLAGS.output_dir, "dev_meta.json")
    num_dev_examples = get_record_num(dev_meta)

    test_meta = os.path.join(FLAGS.output_dir, "test_meta.json")
    num_test_examples = get_record_num(test_meta)
    num_train_steps = int(
        num_train_examples / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model = build_estimator(bert_config, label_list, num_train_steps, num_warmup_steps)

    if FLAGS.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", num_train_examples)
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

        if FLAGS.metric_type not in metric_type.keys():
            raise Exception('metric type {} not be supported'.format(FLAGS.metric_type))

        tf.logging.info('Implement {} type'.format(FLAGS.metric_type))
        metric_func = metric_type[FLAGS.metric_type]

        # exporter = BestExporter(
        #     name="best_exporter",
        #     serving_input_receiver_fn=serving_input_receiver_fn(FLAGS.query_max_seq_length, FLAGS.desc_max_seq_length)(),
        #     compare_fn=metric_func,
        #     exports_to_keep=3
        # )

        exporter = BestCheckpointExporter(
            checkpoint_dir=os.path.join(FLAGS.model_dir, 'best_exporter_checkpoint'),
            name="best_exporter",
            serving_input_receiver_fn=serving_input_receiver_fn(FLAGS.query_max_seq_length, FLAGS.desc_max_seq_length)(),
            compare_fn=metric_func,
            exports_to_keep=3
        )

        train_spec = tf.estimator.TrainSpec(
            input_fn=file_based_input_fn_builder(train_file, FLAGS.query_max_seq_length, FLAGS.desc_max_seq_length, is_training=True,
                                                    drop_remainder=True, batch_size=FLAGS.train_batch_size),
            max_steps=num_train_steps)
        # hooks=[early_stopping])
        eval_spec = tf.estimator.EvalSpec(
            input_fn=file_based_input_fn_builder(dev_file, FLAGS.query_max_seq_length, FLAGS.desc_max_seq_length, is_training=False,
                                                    drop_remainder=False, batch_size=FLAGS.eval_batch_size),
            exporters=exporter,
            steps=None,
            start_delay_secs=FLAGS.start_delay_secs, throttle_secs=FLAGS.throttle_secs)
        tf.estimator.train_and_evaluate(
            estimator=model,
            train_spec=train_spec,
            eval_spec=eval_spec
        )


        # model.train(input_fn=file_based_input_fn_buildertrain_file, FLAGS.max_seq_length, is_training=True,
        #                                             drop_remainder=True, batch_size=FLAGS.train_batch_size),
        #             max_steps=num_train_steps
        #             )



        tf.logging.info('Finish do train...')

    if FLAGS.do_eval:
        tf.logging.info('Start to do eval...')
        model.evaluate(input_fn=file_based_input_fn_builder(dev_file, FLAGS.query_max_seq_length, FLAGS.desc_max_seq_length, is_training=False,
                                                               drop_remainder=False, batch_size=FLAGS.eval_batch_size),
                       checkpoint_path=FLAGS.init_checkpoint)
        tf.logging.info('Finish do eval...')

    if FLAGS.do_predict:
        tf.logging.info('Start to do predict...')
        num_actual_predict_examples = num_test_examples
        result = model.predict(
            input_fn=file_based_input_fn_builder(test_file, FLAGS.query_max_seq_length, FLAGS.desc_max_seq_length, is_training=False,
                                                    drop_remainder=False, batch_size=FLAGS.predict_batch_size),
            checkpoint_path=FLAGS.init_checkpoint)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                if i >= num_actual_predict_examples:
                    break
                output_line = "\t".join(
                    str(class_probability)
                    for class_probability in probabilities) + "\n"
                writer.write(output_line)
                num_written_lines += 1
        tf.logging.info('Finish do predict...')

    if FLAGS.do_pb:
        do_convert_pb()



if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
