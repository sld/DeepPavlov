# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
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

import json
from typing import List, Tuple
import math
from logging import getLogger

import numpy as np
import tensorflow as tf

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.layers.tf_layers import cudnn_bi_lstm, cudnn_bi_gru, bi_rnn
from deeppavlov.core.layers.tf_layers import variational_dropout, INITIALIZER
from deeppavlov.core.models.lr_scheduled_tf_model import LRScheduledTFModel
from deeppavlov.models.seq2seq_go_bot.kb_attn_layer import KBAttention

log = getLogger(__name__)


@register("seq2seq_go_bot_nn")
class Seq2SeqGoalOrientedBotNetwork(LRScheduledTFModel):
    """
    The :class:`~deeppavlov.models.seq2seq_go_bot.bot.GoalOrientedBotNetwork`
    is a recurrent network that encodes user utterance and generates response
    in a sequence-to-sequence manner.

    For network architecture is similar to https://arxiv.org/abs/1705.05414 .

    Parameters:
        hidden_size: RNN hidden layer size.
        target_vocab_size: size of a vocabulary of decoder tokens.
        target_start_of_sequence_index: index of a start of sequence token during
            decoding.
        target_end_of_sequence_index: index of an end of sequence token during decoding.
        knowledge_base_size: number of knowledge base entries.
        kb_attention_hidden_sizes: list of sizes for attention hidden units.
        decoder_embeddings: matrix with embeddings for decoder output tokens, size is
            (`targer_vocab_size` + number of knowledge base entries, embedding size).
        cell_type: type of cell to use as basic unit in encoder.
        intent_feature_size:
        encoder_use_cudnn: boolean indicating whether to use cudnn computed encoder or not
            (cudnn version is faster on gpu).
        encoder_agg_method:
        beam_width: width of beam search decoding.
        l2_regs: l2 regularization weight for decoder.
        dropout_rate: probability of weights' dropout.
        state_dropout_rate: probability of rnn state dropout.
        **kwargs: parameters passed to a parent
            :class:`~deeppavlov.core.models.tf_model.TFModel` class.
    """
    #TODO: update param descriptions

    GRAPH_PARAMS = ['hidden_size', 'knowledge_base_size', 'target_vocab_size',
                    'embedding_size', 'intent_feature_size',
                    'db_feature_size',
                    'encoder_agg_method', 'encoder_agg_size',
                    'kb_embedding_control_sum', 'kb_attention_hidden_sizes',
                    'cell_type']

    def __init__(self,
                 hidden_size: int,
                 target_vocab_size: int,
                 target_start_of_sequence_index: int,
                 target_end_of_sequence_index: int,
                 decoder_embeddings: np.ndarray,
                 knowledge_base_entry_embeddings: np.ndarray = [[]],
                 kb_attention_hidden_sizes: List[int] = [],
                 cell_type: str = 'lstm',
                 intent_feature_size: int = 0,
                 db_feature_size: int = 0,
                 encoder_use_cudnn: bool = False,
                 encoder_agg_method: str = "sum",
                 beam_width: int = 1,
                 l2_regs: List[float] = [0.],
                 dropout_rate: float = 0.0,
                 state_dropout_rate: float = 0.0,
                 **kwargs) -> None:

        # initialize knowledge base embeddings
        self.kb_embedding = np.array(knowledge_base_entry_embeddings)
        if self.kb_embedding.shape[1] > 0:
            self.kb_size = self.kb_embedding.shape[0]
            log.debug("recieved knowledge_base_entry_embeddings with shape = {}"
                      .format(self.kb_embedding.shape))
        else:
            self.kb_size = 0
        # initialize decoder embeddings
        self.decoder_embedding = np.array(decoder_embeddings)
        if self.kb_size > 0:
            if self.kb_embedding.shape[1] != self.decoder_embedding.shape[1]:
                raise ValueError("decoder embeddings should have the same dimension"
                                 " as knowledge base entries' embeddings")
        super().__init__(**kwargs)

        # specify model options
        encoder_agg_size = hidden_size
        if encoder_agg_method == 'concat':
            encoder_agg_size = 2 * hidden_size
        self.opt = {
            'hidden_size': hidden_size,
            'target_vocab_size': target_vocab_size,
            'target_start_of_sequence_index': target_start_of_sequence_index,
            'target_end_of_sequence_index': target_end_of_sequence_index,
            'kb_attention_hidden_sizes': kb_attention_hidden_sizes,
            'kb_embedding_control_sum': float(np.sum(self.kb_embedding)),
            'cell_type': cell_type,
            'intent_feature_size': int(intent_feature_size or 0),
            'db_feature_size': int(db_feature_size or 0),
            'encoder_use_cudnn': encoder_use_cudnn,
            'encoder_agg_method': encoder_agg_method,
            'encoder_agg_size': encoder_agg_size,
            'knowledge_base_size': self.kb_size,
            'embedding_size': self.decoder_embedding.shape[1],
            'beam_width': beam_width,
            'l2_regs': l2_regs,
            'dropout_rate': dropout_rate,
            'state_dropout_rate': state_dropout_rate
        }

        print(self.opt)
        # initialize other parameters
        self._init_params()
        # build computational graph
        self._build_graph()
        # initialize session
        self.sess = tf.Session()
        # from tensorflow.python import debug as tf_debug
        # self.sess = tf_debug.TensorBoardDebugWrapperSession(self.sess, "vimary-pc:7019")

        self.sess.run(tf.global_variables_initializer())

        if tf.train.checkpoint_exists(str(self.load_path.resolve())):
            log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
            self.load()
        else:
            log.info("[initializing `{}` from scratch]".format(self.__class__.__name__))

    def _init_params(self):
        self.hidden_size = self.opt['hidden_size']
        self.tgt_vocab_size = self.opt['target_vocab_size']
        self.tgt_sos_id = self.opt['target_start_of_sequence_index']
        self.tgt_eos_id = self.opt['target_end_of_sequence_index']
        self.kb_attn_hidden_sizes = self.opt['kb_attention_hidden_sizes']
        self.embedding_size = self.opt['embedding_size']
        self.cell_type = self.opt['cell_type'].lower()
        self.intent_feature_size = self.opt['intent_feature_size']
        self.db_feature_size = self.opt['db_feature_size']
        self.encoder_use_cudnn = self.opt['encoder_use_cudnn']
        self.encoder_agg_size = self.opt['encoder_agg_size']
        self.encoder_agg_method = self.opt['encoder_agg_method']
        self.beam_width = self.opt['beam_width']
        self.dropout_rate = self.opt['dropout_rate']
        self.state_dropout_rate = self.opt['state_dropout_rate']
        self.l2_regs = self.opt['l2_regs']

    def _build_graph(self):
        self._add_placeholders()

        self._build_encoder(scope="Encoder")
        self._dec_logits, self._dec_preds = self._build_decoder(scope="Decoder")

        self._dec_loss = self._build_dec_loss(self._dec_logits,
                                              weights=self._tgt_mask,
                                              scopes=["Encoder", "Decoder"],
                                              l2_reg=self.l2_regs[0])

        self._loss = self._dec_loss

        self._train_op = self.get_train_op(self._loss, clip_norm=10)

        log.info("Trainable variables")
        for v in tf.trainable_variables():
            log.info(v)
        self.print_number_of_parameters()

    def _build_dec_loss(self, logits, weights, scopes=[None], l2_reg=0.0):
        # _loss_tensor: [batch_size, max_output_time]
        _loss_tensor = \
            tf.losses.sparse_softmax_cross_entropy(logits=logits,
                                                   labels=self._decoder_outputs,
                                                   weights=tf.expand_dims(weights, -1),
                                                   reduction=tf.losses.Reduction.NONE)
        # check if loss has nans
        _loss_tensor = \
            tf.verify_tensor_all_finite(_loss_tensor, "Non finite values in loss tensor.")
        # _loss: [1]
        _loss = tf.reduce_sum(_loss_tensor) / tf.reduce_sum(weights)
        # add l2 regularization
        if l2_reg > 0:
            reg_vars = [tf.losses.get_regularization_loss(scope=sc, name=f"{sc}_reg_loss")
                        for sc in scopes]
            _loss += l2_reg * tf.reduce_sum(reg_vars)
        return _loss

    def _add_placeholders(self):
        self._dropout_keep_prob = \
            tf.placeholder_with_default(1.0, shape=[], name='dropout_keep_prob')
        self._state_dropout_keep_prob = \
            tf.placeholder_with_default(1.0, shape=[], name='state_dropout_keep_prob')
        # _encoder_inputs: [batch_size, max_input_time, embedding_size]
        self._encoder_inputs = tf.placeholder(tf.float32,
                                              [None, None, self.embedding_size],
                                              name='encoder_inputs')
        self._batch_size = tf.shape(self._encoder_inputs)[0]
        # _decoder_inputs: [batch_size, max_output_time]
        self._decoder_inputs = tf.placeholder(tf.int32,
                                              [None, None],
                                              name='decoder_inputs')
        # _intent_feats: [batch_size, intent_feat_size]
        self._intent_feats = tf.placeholder(tf.float32,
                                            [None, self.intent_feature_size],
                                            name='intent_features')

        self._db_pointer = tf.placeholder(tf.float32,
                                          [None, self.db_feature_size],
                                          name='db_features')
        # _decoder_embedding: [tgt_vocab_size + kb_size, embedding_size]
        # TODO: try training decoder embeddings
        self._decoder_embedding = \
            tf.get_variable("decoder_embedding",
                            shape=(self.tgt_vocab_size + self.kb_size,
                                   self.embedding_size),
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(self.decoder_embedding),
                            trainable=False)
        # _decoder_outputs: [batch_size, max_output_time]
        self._decoder_outputs = tf.placeholder(tf.int32,
                                               [None, None],
                                               name='decoder_outputs')
        # _kb_embedding: [kb_size, embedding_size]
        kb_W = np.array(self.kb_embedding)[:, :self.embedding_size]
        self._kb_embedding = tf.get_variable("kb_embedding",
                                             shape=(kb_W.shape[0], kb_W.shape[1]),
                                             dtype=tf.float32,
                                             initializer=tf.constant_initializer(kb_W),
                                             trainable=True)
        # _kb_mask: [batch_size, kb_size]
        self._kb_mask = tf.placeholder(tf.float32, [None, None], name='kb_mask')

        # _tgt_mask: [batch_size, max_output_time]
        self._tgt_mask = tf.placeholder(tf.float32, [None, None], name='target_weights')
        # _src_sequence_lengths, _tgt_sequence_lengths: [batch_size]
        self._src_sequence_lengths = tf.placeholder(tf.int32,
                                                    [None],
                                                    name='input_sequence_length')
        self._tgt_sequence_lengths = tf.to_int32(tf.reduce_sum(self._tgt_mask, axis=1))

    def _build_encoder(self, scope="Encoder"):
        with tf.variable_scope(scope):
            # _units: [batch_size, max_input_time, embedding_size]
            _units = self._encoder_inputs
            _units = variational_dropout(_units,
                                         self._dropout_keep_prob,
                                         fixed_mask_dims=[1])
            # _units = tf.nn.dropout(_units, self._dropout_keep_prob)

            # _outputs: [2, batch_size, max_input_time, hidden_size]
            # _state: [2, batch_size, hidden_size]
            if self.encoder_use_cudnn:
                if (self.l2_regs[0] > 0) or (self.l2_regs[1] > 0):
                    log.warning("cuDNN RNN are not l2 regularizable")
                if self.cell_type == 'lstm':
                    _outputs, _state = cudnn_bi_lstm(_units,
                                                     self.hidden_size,
                                                     self._src_sequence_lengths)
                elif self.cell_type == 'gru':
                    _outputs, _state = cudnn_bi_gru(_units,
                                                    self.hidden_size,
                                                    self._src_sequence_lengths)
            else:
                _outputs, _state = bi_rnn(_units,
                                          self.hidden_size,
                                          cell_type=self.cell_type,
                                          seq_lengths=self._src_sequence_lengths)

            # _outputs: [batch_size, max_input_time, aggregation_size]
            _outputs = self._aggregate_encoder_outs(_outputs)
            # _state: [batch_size, max_input_time, hidden_size]
            if (self.cell_type == 'lstm') and\
                    not isinstance(_state[0], tf.nn.rnn_cell.LSTMStateTuple):
                _state_c = self._aggregate_encoder_outs([_state[0][0],
                                                         _state[1][0]])
                _state_h = self._aggregate_encoder_outs([_state[0][1],
                                                         _state[1][1]])
                _state = self._build_intent(_state_c, self._intent_feats, self._db_pointer)
            else:
                _state = self._aggregate_encoder_outs(_state)
                _state = self._build_intent(_state, self._intent_feats, self._db_pointer)

            # TODO: add & validate cell dropout
            # NOTE: not available for CUDNN cells?

        self._encoder_outputs = _outputs
        self._intent = _state

    def _aggregate_encoder_outs(self,
                                outs: tf.Tensor,
                                scope: str = "") -> tf.Tensor:
        """ Aggregates encoder outputs along last axis"""
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if self.encoder_agg_method == "concat":
                # outs: [..., 2 * last_size]
                outs = tf.concat(outs, -1)
            elif self.encoder_agg_method == "weight_sum":
                # outs: [... , last_size]
                _w_init = tf.constant_initializer([1.0, -1.0])
                _weights = tf.get_variable('encoder_agg_weights',
                                           [2],
                                           initializer=_w_init,
                                           trainable=True)
                _gain = tf.get_variable('encoder_agg_gain',
                                        [],
                                        initializer=tf.ones_initializer(),
                                        trainable=True)
                _weights_sm = tf.nn.softmax(_weights)

                outs = tf.stack(outs, -1)
                outs = tf.reduce_sum(outs * _weights_sm, -1) * _gain
                self._weights, self._gain = _weights, _gain
            else:
                # outs: [..., last_size]
                outs = tf.stack(outs, -1)
                outs = tf.reduce_sum(outs, -1)
        return outs

    def _build_intent(self, enc_feats, intent_features, db_features, scope="Intent"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            _enc_weights = tf.get_variable("encoder_weights",
                                           (self.encoder_agg_size,
                                            self.hidden_size),
                                           initializer=tf.truncated_normal_initializer(stddev=0.2))
            _intent_weights = tf.get_variable("intent_weights",
                                           (self.intent_feature_size,
                                            self.hidden_size),
                                           initializer=tf.truncated_normal_initializer(stddev=0.2))
            _db_weights = tf.get_variable("db_weights",
                                           (self.db_feature_size,
                                            self.hidden_size),
                                           initializer=tf.truncated_normal_initializer(stddev=0.2))
            output = tf.matmul(enc_feats, _enc_weights) + tf.matmul(intent_features, _intent_weights) + tf.matmul(db_features, _db_weights)
            output = tf.tanh(output)
        return output

    def _build_decoder(self, scope="Decoder"):
        with tf.variable_scope(scope):
            # Decoder embedding
            _decoder_emb_inp = tf.nn.embedding_lookup(self._decoder_embedding,
                                                      self._decoder_inputs)

            # Tiling outputs, states, sequence lengths
            _tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
                self._encoder_outputs, multiplier=self.beam_width)
            _tiled_intent = tf.contrib.seq2seq.tile_batch(
                self._intent, multiplier=self.beam_width)
            _tiled_src_sequence_lengths = tf.contrib.seq2seq.tile_batch(
                self._src_sequence_lengths, multiplier=self.beam_width)

            if self.kb_size > 0:
                with tf.variable_scope("AttentionOverKB"):
                    _projection_layer = KBAttention(self.tgt_vocab_size,
                                                    self.kb_attn_hidden_sizes + [1],
                                                    self._kb_embedding,
                                                    self._kb_mask,
                                                    activation=tf.nn.relu,
                                                    use_bias=False)
            else:
                with tf.variable_scope("OutputDense"):
                    # TODO: PUT DB FEATS IN HERE!!!
                    _projection_layer = DenseWithConcat(self.tgt_vocab_size, self._intent_feats, self._db_pointer)
                    # _projection_layer = tf.layers.Dense(self.tgt_vocab_size,
                    #                                     use_bias=False)

            def _build_step_fn(memory, memory_seq_len, init_state, scope, reuse=None):
                with tf.variable_scope("decode_with_shared_attention", reuse=reuse):
                    # Decoder Cell
                    if self.cell_type.lower() == 'lstm':
                        _cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size,
                                                        initializer=INITIALIZER(),
                                                        name='basic_lstm_cell')
                    elif self.cell_type.lower() == 'gru':
                        _cell = tf.nn.rnn_cell.GRUCell(self.hidden_size,
                                                       kernel_initializer=INITIALIZER(),
                                                       name='basic_gru_cell')

                    # Checking init_state shape
                    _batch_size = tf.shape(memory)[0]
                    # TODO: try placing dropout after attention
                    _cell = tf.contrib.rnn.DropoutWrapper(
                        _cell,
                        input_size=self.embedding_size + self.encoder_agg_size,
                        dtype=memory.dtype,
                        input_keep_prob=self._dropout_keep_prob,
                        output_keep_prob=self._dropout_keep_prob,
                        state_keep_prob=self._state_dropout_keep_prob,
                        variational_recurrent=True)

                    # Attention mechanism
                    # _attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    _attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                        self.hidden_size,
                        memory=memory,
                        memory_sequence_length=memory_seq_len)

                    _cell = tf.contrib.seq2seq.AttentionWrapper(
                        _cell,
                        _attention_mechanism,
                        attention_layer_size=self.encoder_agg_size,
                        alignment_history=False,
                        initial_cell_state=init_state)

                    def _fn(step, inputs, state):
                        # This scope is defined by tf.contrib.seq2seq.dynamic_decode
                        # during the training.
                        with tf.variable_scope("decoder", reuse=reuse):
                            outputs, state = _cell(inputs, state)
                            return outputs, state

                    _init_state = _cell.zero_state(_batch_size, dtype=memory.dtype)

                    return _fn, _cell, _init_state

            # TRAIN MODE
            _, _decoder_tr_cell, _decoder_tr_init_state = \
                _build_step_fn(memory=self._encoder_outputs,
                               memory_seq_len=self._src_sequence_lengths,
                               init_state=self._intent,
                               scope="dec_cell_step")

            # Train Helper to feed inputs for training:
            # read inputs from dense ground truth vectors
            _helper_tr = tf.contrib.seq2seq.TrainingHelper(
                _decoder_emb_inp, self._tgt_sequence_lengths, time_major=False)

            # Copy encoder hidden state to decoder inital state
            _decoder_init_state = \
                _decoder_tr_cell.zero_state(self._batch_size, dtype=tf.float32)\
                .clone(cell_state=self._intent)

            # TODO: debug beam search: wrong states?
            _decoder_tr = \
                tf.contrib.seq2seq.BasicDecoder(_decoder_tr_cell,
                                                _helper_tr,
                                                initial_state=_decoder_tr_init_state,
                                                output_layer=_projection_layer)

            # Wrap into variable scope to share attention parameters
            # Required!
            with tf.variable_scope("decode_with_shared_attention"):
                _outputs_tr, _, _ = \
                    tf.contrib.seq2seq.dynamic_decode(_decoder_tr,
                                                      impute_finished=False,
                                                      output_time_major=False)
                _logits = _outputs_tr.rnn_output

            # INFER MODE
            _, _decoder_inf_cell, _decoder_inf_init_state = \
                _build_step_fn(memory=_tiled_encoder_outputs,
                               memory_seq_len=_tiled_src_sequence_lengths,
                               init_state=_tiled_intent,
                               scope="dec_cell_step",
                               reuse=True)
            _max_iters = tf.round(tf.reduce_max(self._src_sequence_lengths) * 2)
            # NOTE: helper is not needed?
            # _helper_inf = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            #    self._decoder_embedding,
            #    tf.fill([self._batch_size], self.tgt_sos_id), self.tgt_eos_id)
            #    lambda d: tf.one_hot(d, self.tgt_vocab_size + self.kb_size),
            # Define a beam-search decoder
            _start_tokens = tf.tile(tf.constant([self.tgt_sos_id], tf.int32),
                                    [self._batch_size])

            _decoder_inf = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=_decoder_inf_cell,
                    embedding=self._decoder_embedding,
                    start_tokens=_start_tokens,
                    end_token=self.tgt_eos_id,
                    initial_state=_decoder_inf_init_state,
                    beam_width=self.beam_width,
                    output_layer=_projection_layer,
                    length_penalty_weight=0.0)

            # Wrap into variable scope to share attention parameters
            # Required!
            with tf.variable_scope("decode_with_shared_attention", reuse=True):
                _outputs_inf, _, _ =\
                    tf.contrib.seq2seq.dynamic_decode(_decoder_inf,
                                                      impute_finished=False,
                                                      maximum_iterations=_max_iters,
                                                      output_time_major=False)
                _predictions = _outputs_inf.predicted_ids[:, :, 0]
        return _logits, _predictions

    def __call__(self, enc_inputs, src_seq_lens, intent_feats, kb_masks, db_pointer,
                 prob=False):
        dec_preds = self.sess.run(
            self._dec_preds,
            feed_dict={
                self._dropout_keep_prob: 1.,
                self._state_dropout_keep_prob: 1.,
                self._encoder_inputs: enc_inputs,
                self._src_sequence_lengths: src_seq_lens,
                self._intent_feats: intent_feats,
                self._kb_mask: kb_masks,
                self._db_pointer: db_pointer
            }
        )
# TODO: implement infer probabilities
        if prob:
            raise NotImplementedError("Probs not available for now.")
        return dec_preds

    def train_on_batch(self, enc_inputs, dec_inputs, dec_outputs, src_seq_lens,
                       tgt_masks, intent_feats, kb_masks, db_pointer):
        _, loss, dec_loss = self.sess.run(
            [self._train_op, self._loss, self._dec_loss],
            feed_dict={
                self._dropout_keep_prob: 1 - self.dropout_rate,
                self._state_dropout_keep_prob: 1 - self.state_dropout_rate,
                self._encoder_inputs: enc_inputs,
                self._decoder_inputs: dec_inputs,
                self._decoder_outputs: dec_outputs,
                self._src_sequence_lengths: src_seq_lens,
                self._tgt_mask: tgt_masks,
                self._intent_feats: intent_feats,
                self._kb_mask: kb_masks,
                self._db_pointer: db_pointer
            }
        )
        return {'loss': loss,
                'learning_rate': self.get_learning_rate(),
                'momentum': self.get_momentum(),
                'last_dec_loss': dec_loss}

    def load(self, *args, **kwargs):
        self.load_params()
        super().load(*args, **kwargs)

    def load_params(self):
        path = str(self.load_path.with_suffix('.json').resolve())
        log.info('[loading parameters from {}]'.format(path))
        with open(path, 'r', encoding='utf8') as fp:
            params = json.load(fp)
        for p in self.GRAPH_PARAMS:
            if self.opt.get(p) != params.get(p):
                if p in ('kb_embedding_control_sum') and\
                        (math.abs(self.opt.get(p, 0.) - params.get(p, 0.)) < 1e-3):
                        continue
                raise ConfigError("`{}` parameter must be equal to saved model"
                                  " parameter value `{}`, but is equal to `{}`"
                                  .format(p, params.get(p), self.opt.get(p)))

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        self.save_params()

    def save_params(self):
        path = str(self.save_path.with_suffix('.json').resolve())
        log.info('[saving parameters to {}]'.format(path))
        with open(path, 'w', encoding='utf8') as fp:
            json.dump(self.opt, fp)


@register("seq2seq_go_bot_with_ner_nn")
class Seq2SeqGoalOrientedBotWithNerNetwork(Seq2SeqGoalOrientedBotNetwork):
    """
    The :class:`~deeppavlov.models.seq2seq_go_bot.bot.GoalOrientedBotNetwork`
    is a recurrent network that encodes user utterance and generates response
    in a sequence-to-sequence manner.

    For network architecture is similar to https://arxiv.org/abs/1705.05414 .

    Parameters:
        ner_n_tags: number of classifiered tags.
        ner_beta: multiplier of ner loss in the overall loss
        ner_hidden_size: list of integers denoting hidden sizes of ner head.
        hidden_size: RNN hidden layer size.
        target_vocab_size: size of a vocabulary of decoder tokens.
        target_start_of_sequence_index: index of a start of sequence token during
            decoding.
        target_end_of_sequence_index: index of an end of sequence token during decoding.
        knowledge_base_size: number of knowledge base entries.
        kb_attention_hidden_sizes: list of sizes for attention hidden units.
        decoder_embeddings: matrix with embeddings for decoder output tokens, size is
            (`targer_vocab_size` + number of knowledge base entries, embedding size).
        cell_type: type of cell to use as basic unit in encoder.
        intent_feature_size:
        db_feature_size:
        encoder_use_cudnn: boolean indicating whether to use cudnn computed encoder or not
            (cudnn version is faster on gpu).
        encoder_agg_method:
        beam_width: width of beam search decoding.
        l2_regs: tuple of l2 regularization weights for decoder and ner losses.
        dropout_rate: probability of weights' dropout.
        state_dropout_rate: probability of rnn state dropout.
        **kwargs: parameters passed to a parent
            :class:`~deeppavlov.core.models.tf_model.TFModel` class.
    """

    GRAPH_PARAMS = ['ner_n_tags', 'ner_hidden_size', 'hidden_size',
                    'knowledge_base_size', 'target_vocab_size', 'embedding_size',
                    'kb_embedding_control_sum', 'kb_attention_hidden_sizes',
                    'cell_type', 'encoder_agg_method', 'encoder_agg_size',
                    'intent_feature_size']

    def __init__(self,
                 ner_n_tags: int,
                 ner_beta: float,
                 hidden_size: int,
                 target_vocab_size: int,
                 target_start_of_sequence_index: int,
                 target_end_of_sequence_index: int,
                 decoder_embeddings: np.ndarray,
                 ner_hidden_size: List[int] = [],
                 knowledge_base_entry_embeddings: np.ndarray = [[]],
                 kb_attention_hidden_sizes: List[int] = [],
                 cell_type: str = 'lstm',
                 intent_feature_size: int = 0,
                 encoder_use_cudnn: bool = False,
                 encoder_agg_method: str = "sum",
                 beam_width: int = 1,
                 l2_regs: Tuple[float, float] = [0., 0.],
                 dropout_rate: float = 0.0,
                 state_dropout_rate: float = 0.0,
                 **kwargs) -> None:

        # initialize knowledge base embeddings
        self.kb_embedding = np.array(knowledge_base_entry_embeddings)
        if self.kb_embedding.shape[1] > 0:
            self.kb_size = self.kb_embedding.shape[0]
            log.debug("recieved knowledge_base_entry_embeddings with shape = {}"
                      .format(self.kb_embedding.shape))
        else:
            self.kb_size = 0
        # initialize decoder embeddings
        self.decoder_embedding = np.array(decoder_embeddings)
        if self.kb_size > 0:
            if self.kb_embedding.shape[1] != self.decoder_embedding.shape[1]:
                raise ValueError("decoder embeddings should have the same dimension"
                                 " as knowledge base entries' embeddings")
        super(Seq2SeqGoalOrientedBotNetwork, self).__init__(**kwargs)

        # specify model options
        encoder_agg_size = hidden_size
        if encoder_agg_method == 'concat':
            encoder_agg_size = 2 * hidden_size
        self.opt = {
            'ner_n_tags': ner_n_tags,
            'ner_hidden_size': ner_hidden_size,
            'ner_beta': ner_beta,
            'hidden_size': hidden_size,
            'target_vocab_size': target_vocab_size,
            'target_start_of_sequence_index': target_start_of_sequence_index,
            'target_end_of_sequence_index': target_end_of_sequence_index,
            'kb_attention_hidden_sizes': kb_attention_hidden_sizes,
            'kb_embedding_control_sum': float(np.sum(self.kb_embedding)),
            'cell_type': cell_type,
            'intent_feature_size': int(intent_feature_size or 0),
            'encoder_use_cudnn': encoder_use_cudnn,
            'encoder_agg_method': encoder_agg_method,
            'encoder_agg_size': encoder_agg_size,
            'knowledge_base_size': self.kb_size,
            'embedding_size': self.decoder_embedding.shape[1],
            'beam_width': beam_width,
            'l2_regs': l2_regs,
            'dropout_rate': dropout_rate,
            'state_dropout_rate': state_dropout_rate
        }

        # initialize other parameters
        self._init_params()
        # build computational graph
        self._build_graph()
        # initialize session
        self.sess = tf.Session()
        # from tensorflow.python import debug as tf_debug
        # self.sess = tf_debug.TensorBoardDebugWrapperSession(self.sess, "vimary-pc:7019")

        self.sess.run(tf.global_variables_initializer())

        if tf.train.checkpoint_exists(str(self.load_path.resolve())):
            log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
            self.load()
        else:
            log.info("[initializing `{}` from scratch]".format(self.__class__.__name__))

    def _init_params(self):
        super()._init_params()
        self.ner_n_tags = self.opt['ner_n_tags']
        self.ner_hidden_size = self.opt['ner_hidden_size']
        self.ner_beta = self.opt['ner_beta']

        if len(self.opt['l2_regs']) != 2:
            raise ConfigError("`l2_regs` parameter should be a tuple two floats.")
        self.l2_regs = self.opt['l2_regs']

    def _build_graph(self):
        self._add_placeholders()

        self._build_encoder(scope="Encoder")
        self._dec_logits, self._dec_preds = self._build_decoder(scope="Decoder")
        self._ner_logits = self._build_ner_head(scope="NerHead")

        self._dec_loss = self._build_dec_loss(self._dec_logits,
                                              weights=self._tgt_mask,
                                              scopes=["Encoder", "Decoder"],
                                              l2_reg=self.l2_regs[0])

        self._ner_loss, self._ner_preds = \
            self._build_ner_loss_predict(self._ner_logits,
                                         weights=self._src_tag_mask,
                                         n_tags=self.ner_n_tags,
                                         scopes=["NerHead"],
                                         l2_reg=self.l2_regs[1])

        self._loss = self._dec_loss + self.ner_beta * self._ner_loss

        self._train_op = self.get_train_op(self._loss, clip_norm=10)

        log.info("Trainable variables")
        for v in tf.trainable_variables():
            log.info(v)
        self.print_number_of_parameters()

    def _build_ner_loss_predict(self, logits, weights, n_tags, scopes=[None], l2_reg=0.0):
        # _loss_tensor: [batch_size, max_input_time]
        _loss_tensor = \
            tf.losses.sparse_softmax_cross_entropy(logits=logits,
                                                   labels=self._src_tags,
                                                   weights=tf.expand_dims(weights, -1),
                                                   reduction=tf.losses.Reduction.NONE)
        # check if loss has nans
        _loss_tensor = \
            tf.verify_tensor_all_finite(_loss_tensor, "Non finite values in loss tensor.")
        # _loss: [1]
        _loss = tf.reduce_sum(_loss_tensor) / tf.reduce_sum(weights)
        # add l2 regularization
        if l2_reg > 0:
            reg_vars = [tf.losses.get_regularization_loss(scope=sc, name=f"{sc}_reg_loss")
                        for sc in scopes]
            _loss += l2_reg * tf.reduce_sum(reg_vars)

        # _preds: [batch_size, max_input_time]
        _preds = tf.argmax(logits, axis=-1)
        return _loss, _preds

    def _add_placeholders(self):
        super()._add_placeholders()
        # _src_mask: [batch_size, max_input_time]
        self._src_tag_mask = tf.placeholder(tf.float32,
                                            [None, None],
                                            name='input_sequence_tag_mask')
        # _src_tags: [batch_size, max_input_time]
        self._src_tags = tf.placeholder(tf.int32,
                                        [None, None],
                                        name='input_sequence_tags')

    def _build_ner_head(self, scope="NerHead"):
        with tf.variable_scope(scope):
            # _units: [batch_size, max_input_time, encoder_agg_size]
            _units = self._encoder_outputs

            # TODO: try dropout
            # _units = variational_dropout(_units,
            #                             self._dropout_keep_prob,
            #                             fixed_mask_dims=[1])
            _units = tf.nn.dropout(_units, 0.9)
            for n_hidden in self.ner_hidden_size:
                # _units: [batch_size, max_input_time, n_hidden]
                _units = tf.layers.dense(_units, n_hidden, activation=tf.nn.relu,
                                         kernel_initializer=INITIALIZER(),
                                         kernel_regularizer=tf.nn.l2_loss)
            # _ner_logits: [batch_size, max_input_time, ner_n_tags]
            _logits = tf.layers.dense(_units, self.ner_n_tags, activation=None,
                                      kernel_initializer=INITIALIZER(),
                                      kernel_regularizer=tf.nn.l2_loss)
            return _logits

    def __call__(self, enc_inputs, src_seq_lens, src_tag_masks, intent_feats,
                 kb_masks, prob=False):
        dec_preds, ner_preds, weights, gain = self.sess.run(
            [self._dec_preds, self._ner_preds, self._weights, self._gain],
            # [self._dec_preds, self._ner_preds],
            [self._dec_preds, self._ner_preds],
            feed_dict={
                self._dropout_keep_prob: 1.,
                self._state_dropout_keep_prob: 1.,
                self._encoder_inputs: enc_inputs,
                self._src_tag_mask: src_tag_masks,
                self._src_sequence_lengths: src_seq_lens,
                self._intent_feats: intent_feats,
                self._kb_mask: kb_masks
            }
        )
        log.info(f"weights = {weights}, gain = {gain}")
        ner_preds_cut = []
        for i in range(len(ner_preds)):
            nonzero_ids = np.nonzero(src_tag_masks[i])[0]
            ner_preds_cut.append(ner_preds[i, nonzero_ids])
# TODO: implement infer probabilities
        if prob:
            raise NotImplementedError("Probs not available for now.")
        return dec_preds, ner_preds_cut

    def train_on_batch(self, enc_inputs, dec_inputs, dec_outputs, src_tags,
                       src_seq_lens, tgt_masks, src_tag_masks, intent_feats,
                       kb_masks):
        _, loss, dec_loss, ner_loss = self.sess.run(
            [self._train_op, self._loss, self._dec_loss, self._ner_loss],
            feed_dict={
                self._dropout_keep_prob: 1 - self.dropout_rate,
                self._state_dropout_keep_prob: 1 - self.state_dropout_rate,
                self._encoder_inputs: enc_inputs,
                self._decoder_inputs: dec_inputs,
                self._decoder_outputs: dec_outputs,
                self._src_tags: src_tags,
                self._src_sequence_lengths: src_seq_lens,
                self._tgt_mask: tgt_masks,
                self._src_tag_mask: src_tag_masks,
                self._intent_feats: intent_feats,
                self._kb_mask: kb_masks
            }
        )
        return {'loss': loss,
                'learning_rate': self.get_learning_rate(),
                'momentum': self.get_momentum(),
                'last_dec_loss': dec_loss,
                'last_ner_loss': ner_loss}


# _projection_layer = tf.layers.Dense(self.tgt_vocab_size, use_bias=False)
class DenseWithConcat(tf.layers.Dense):
  def __init__(self, dim, intent_feats, db_feats, **kwargs):
    super(DenseWithConcat, self).__init__(units=dim, **kwargs)
    self.intent_feats = intent_feats
    self.db_feats = db_feats
    self.dim = dim
    self.dense = tf.layers.Dense(dim, use_bias=False)

  def __call__(self, inputs):
    # TODO: Add tf.print to check if db feats and intent feats hadn't had cached
    if len(inputs.shape) == 3:
        db_feats = tf.reshape(self.db_feats, [-1, 1, 30])
        intent_feats = tf.reshape(self.intent_feats, [-1, 1, 92])
        x = tf.concat([inputs, intent_feats, db_feats], axis=2)
    else:
        db_feats = self.db_feats
        intent_feats = self.intent_feats
        x = tf.concat([inputs, intent_feats, db_feats], axis=1)
    x = self.dense(x)
    return x

