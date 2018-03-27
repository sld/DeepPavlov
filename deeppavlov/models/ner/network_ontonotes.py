"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import tensorflow as tf
from deeppavlov.core.layers.tf_layers import cudnn_bi_lstm
from tensorflow.contrib.layers import xavier_initializer
from nltk.tag import SennaNERTagger, SennaChunkTagger

TAG_DICT = {'B-CARDINAL': 16,
            'B-DATE': 9,
            'B-EVENT': 7,
            'B-FAC': 19,
            'B-GPE': 36,
            'B-LANGUAGE': 11,
            'B-LAW': 12,
            'B-LOC': 8,
            'B-MONEY': 34,
            'B-NORP': 18,
            'B-ORDINAL': 21,
            'B-ORG': 23,
            'B-PERCENT': 15,
            'B-PERSON': 1,
            'B-PRODUCT': 14,
            'B-QUANTITY': 35,
            'B-TIME': 4,
            'B-WORK_OF_ART': 3,
            'I-CARDINAL': 31,
            'I-DATE': 10,
            'I-EVENT': 25,
            'I-FAC': 30,
            'I-GPE': 32,
            'I-LANGUAGE': 24,
            'I-LAW': 29,
            'I-LOC': 0,
            'I-MONEY': 17,
            'I-NORP': 27,
            'I-ORDINAL': 6,
            'I-ORG': 33,
            'I-PERCENT': 2,
            'I-PERSON': 22,
            'I-PRODUCT': 5,
            'I-QUANTITY': 13,
            'I-TIME': 26,
            'I-WORK_OF_ART': 28,
            'O': 20}

NER_DICT = {'B-LOC': 1,
            'B-MISC': 2,
            'B-ORG': 3,
            'B-PER': 4,
            'I-LOC': 5,
            'I-MISC': 6,
            'I-ORG': 7,
            'I-PER': 8,
            'O': 0}

POS_DICT = {'B-ADJP': 3,
            'B-ADVP': 12,
            'B-CONJP': 5,
            'B-INTJ': 15,
            'B-LST': 16,
            'B-NP': 10,
            'B-PP': 1,
            'B-PRT': 9,
            'B-SBAR': 2,
            'B-VP': 4,
            'I-ADJP': 11,
            'I-ADVP': 13,
            'I-CONJP': 14,
            'I-NP': 7,
            'I-PP': 17,
            'I-SBAR': 6,
            'I-VP': 8,
            'O': 0}


class OntoNER:
    def __init__(self,
                 embedder,
                 n_hidden=(256, 256, 256),
                 token_embeddings_dim=100,
                 gpu=0):
        n_tags = len(TAG_DICT)

        # Create placeholders
        x_word = tf.placeholder(dtype=tf.float32, shape=[None, None, token_embeddings_dim], name='x_word')
        x_char = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='x_char')

        # Features
        x_pos = tf.placeholder(dtype=tf.float32, shape=[None, None, len(POS_DICT)], name='x_pos')  # Senna
        x_ner = tf.placeholder(dtype=tf.float32, shape=[None, None, len(NER_DICT)], name='x_ner')  # Senna
        x_capi = tf.placeholder(dtype=tf.float32, shape=[None, None], name='x_capi')

        y_true = tf.placeholder(dtype=tf.int32, shape=[None, None], name='y_tag')
        mask = tf.placeholder(dtype=tf.float32, shape=[None, None], name='mask')
        sequence_lengths = tf.reduce_sum(mask, axis=1)

        # Concat features to embeddings
        emb = tf.concat([x_word, tf.expand_dims(x_capi, 2), x_pos, x_ner], axis=2)

        # The network
        units = emb
        for n, n_h in enumerate(n_hidden):
            with tf.variable_scope('RNN_' + str(n)):
                units, _ = cudnn_bi_lstm(units, n_h, tf.to_int32(sequence_lengths))

        # Classifier
        with tf.variable_scope('Classifier'):
            units = tf.layers.dense(units, n_hidden[-1], kernel_initializer=xavier_initializer())
            logits = tf.layers.dense(units, n_tags, kernel_initializer=xavier_initializer())

        # CRF
        _, trainsition_params = tf.contrib.crf.crf_log_likelihood(logits,
                                                                  y_true,
                                                                  sequence_lengths)

        # Initialize session
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = str(gpu)
        sess = tf.Session(config=config)

        self._ner_tagger = SennaNERTagger('download/senna/')
        self._pos_tagger = SennaChunkTagger('download/senna/')

        self._x_w = x_word
        self._x_c = x_char
        self._x_capi = x_capi
        self.x_pos = x_pos
        self.x_ner = x_ner
        self._y_true = y_true
        self._mask = mask
        self._sequence_lengths = sequence_lengths
        self._token_embeddings_dim = token_embeddings_dim

        self._logits = logits
        self._trainsition_params = trainsition_params

        self._sess = sess
        sess.run(tf.global_variables_initializer())
        self._embedder = embedder
        self.load('download/ner_onto/model.ckpt')

    def load(self, model_file_path):
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(self._sess, model_file_path)

    @staticmethod
    def to_one_hot(x, n):
        b = np.zeros([len(x), n], dtype=np.float32)
        for q, tok in enumerate(x):
            b[q, tok] = 1
        return b

    def tokens_batch_to_numpy_batch(self, batch_x):
        """ Convert a batch of tokens to numpy arrays of features"""
        x = dict()
        batch_size = len(batch_x)
        max_utt_len = max([len(utt) for utt in batch_x])

        # Embeddings
        x['emb'] = np.zeros([batch_size, max_utt_len, self._token_embeddings_dim], dtype=np.float32)
        for n, utterance in enumerate(batch_x):
            for q, token in enumerate(utterance):
                try:
                    x['emb'][n][q] = self._embedder[token.lower()]
                except KeyError:
                    pass

        # Capitalization
        x['capitalization'] = np.zeros([batch_size, max_utt_len], dtype=np.float32)
        for n, utt in enumerate(batch_x):
            x['capitalization'][n, :len(utt)] = [tok[0].isupper() for tok in utt]

        # POS
        n_pos = len(POS_DICT)
        x['pos'] = np.zeros([batch_size, max_utt_len, n_pos])
        for n, utt in enumerate(batch_x):
            token_tag_pairs = self._pos_tagger.tag(utt)
            pos_tags = list(zip(*token_tag_pairs))[1]
            pos = np.array([POS_DICT[p] for p in pos_tags])
            pos = self.to_one_hot(pos, n_pos)
            x['pos'][n, :len(pos)] = pos

        # NER
        n_ner = len(NER_DICT)
        x['ner'] = np.zeros([batch_size, max_utt_len, n_ner])
        for n, utt in enumerate(batch_x):
            token_tag_pairs = self._ner_tagger.tag(utt)
            ner_tags = list(zip(*token_tag_pairs))[1]
            ner = np.array([NER_DICT[p] for p in ner_tags])
            ner = self.to_one_hot(ner, n_ner)
            x['ner'][n, :len(ner)] = ner

        # Mask for paddings
        x['mask'] = np.zeros([batch_size, max_utt_len], dtype=np.float32)
        for n in range(batch_size):
            x['mask'][n, :len(batch_x[n])] = 1

        return x

    def train_on_batch(self, x_word, x_char, y_tag):
        raise NotImplementedError

    def predict(self, x):
        feed_dict = self._fill_feed_dict(x, training=False)
        y_pred = []
        logits, trans_params, sequence_lengths = self._sess.run([self._logits,
                                                                 self._trainsition_params,
                                                                 self._sequence_lengths
                                                                 ],
                                                                feed_dict=feed_dict)

        # iterate over the sentences because no batching in viterbi_decode
        for logit, sequence_length in zip(logits, sequence_lengths):
            logit = logit[:int(sequence_length)]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
            y_pred += [viterbi_seq]

        pred = []
        batch_size = len(x['emb'])
        idx_to_tag = {value: key for key, value in TAG_DICT.items()}
        for n in range(batch_size):
            pred.append([idx_to_tag[tag] for tag in y_pred[n]])
        return pred

    def predict_for_token_batch(self, tokens_batch):
        batch_x = self.tokens_batch_to_numpy_batch(tokens_batch)
        # Prediction indices
        predictions_batch = self.predict(batch_x)
        predictions_batch_no_pad = list()
        for n, predicted_tags in enumerate(predictions_batch):
            predictions_batch_no_pad.append(predicted_tags[: len(tokens_batch[n])])
        return predictions_batch_no_pad

    def _fill_feed_dict(self,
                        x,
                        y_t=None,
                        learning_rate=None,
                        training=False,
                        dropout_rate=1,
                        learning_rate_decay=1):

        feed_dict = dict()
        feed_dict[self._x_w] = x['emb']
        feed_dict[self._mask] = x['mask']

        # POS and NER features
        feed_dict[self.x_pos] = x['pos']
        feed_dict[self.x_ner] = x['ner']

        # Capitalization
        feed_dict[self._x_capi] = x['capitalization']
        return feed_dict
