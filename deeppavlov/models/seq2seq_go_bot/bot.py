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

import itertools
from logging import getLogger
from typing import Dict, List, Tuple, Any

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.models.seq2seq_go_bot.network import Seq2SeqGoalOrientedBotNetwork
from deeppavlov.models.seq2seq_go_bot.network import Seq2SeqGoalOrientedBotWithNerNetwork

log = getLogger()


@register("seq2seq_go_bot")
class Seq2SeqGoalOrientedBot(NNModel):
    """
    A goal-oriented bot based on a sequence-to-sequence rnn. For implementation details see
    :class:`~deeppavlov.models.seq2seq_go_bot.network.Seq2SeqGoalOrientedBotNetwork`.
    Pretrained for :class:`~deeppavlov.dataset_readers.kvret_reader.KvretDatasetReader` dataset.

    Parameters:
        network_parameters: parameters passed to object of
            :class:`~deeppavlov.models.seq2seq_go_bot.network.Seq2SeqGoalOrientedBotNetwork` class.
        embedder: word embeddings model, see
            :doc:`deeppavlov.models.embedders </apiref/models/embedders>`.
        start_of_sequence_token: token that defines start of input sequence.
        end_of_sequence_token: token that defines end of input sequence and start of
            output sequence.
        debug: whether to display debug output.
        **kwargs: parameters passed to parent
            :class:`~deeppavlov.core.models.nn_model.NNModel` class.
    """
    def __init__(self,
                 network_parameters: Dict,
                 embedder: Component,
                 target_vocab: Component,
                 start_of_sequence_token: str,
                 end_of_sequence_token: str,
                 save_path: str,
                 load_path: str = None,
                 delimiters: Tuple[str, str] = (' ', ' '),
                 knowledge_base_keys: List[str] = [],
                 debug: bool = False,
                 **kwargs) -> None:
        super().__init__(save_path=save_path, load_path=load_path, **kwargs)

        self.embedder = embedder
        self.embedding_size = embedder.dim
        self.tgt_vocab = target_vocab
        self.tgt_vocab_size = len(target_vocab)
        self.kb_keys = knowledge_base_keys
        self.kb_size = len(self.kb_keys)
        self.delimiters = delimiters
        self.sos_token = start_of_sequence_token
        self.eos_token = end_of_sequence_token
        self.debug = debug

        network_parameters['load_path'] = load_path
        network_parameters['save_path'] = save_path
        self.use_ner_head = ('ner_n_tags' in network_parameters)
        self.use_state_features = \
            (network_parameters.get('intent_feature_size', 0) > 0)
        self.use_db_features = \
            (network_parameters.get('db_feature_size', 0) > 0)
        self.use_graph_features = \
            (network_parameters.get('graph_feature_size', 0) > 0)
        self.use_graph_ans_features = \
            (network_parameters.get('graph_ans_feature_size', 0) > 0)
        self.use_kb_attention = (self.kb_size != 0)
        self.network = self._init_network(network_parameters, use_ner_head=self.use_ner_head)

    def _init_network(self, params, use_ner_head=False):
        if 'target_start_of_sequence_index' not in params:
            params['target_start_of_sequence_index'] = self.tgt_vocab[self.sos_token]
        if 'target_end_of_sequence_index' not in params:
            params['target_end_of_sequence_index'] = self.tgt_vocab[self.eos_token]
        # construct matrix of knowledge bases values embeddings
        if self.kb_size > 0:
            params['knowledge_base_entry_embeddings'] = \
                [self._embed_kb_key(val) for val in self.kb_keys]
        # construct matrix of decoder input token embeddings (zeros for sos_token)
        dec_embs = self.embedder([[self.tgt_vocab[idx]
                                   for idx in range(self.tgt_vocab_size)]])[0]
        dec_embs[self.tgt_vocab[self.sos_token]][:] = 0.
        params['decoder_embeddings'] = dec_embs
        if use_ner_head:
            return Seq2SeqGoalOrientedBotWithNerNetwork(**params)
        return Seq2SeqGoalOrientedBotNetwork(**params)

    def _embed_kb_key(self, key):
# TODO: fasttext embedder to work with tokens
        emb = np.array(self.embedder([key.split('_')], mean=True)[0])
        if self.debug:
            log.debug("embedding key tokens='{}', embedding shape = {}"
                      .format(key.split('_'), emb.shape))
        return emb

    def fit(self, *args):
        raise NotImplementedError("`fit_on` not implemented yet")
        data = [self.preprocess(*([xi] for xi in x)) for x in zip(*args)]
        return self.network.fit(*list(zip(*data)))

    def preprocess(self, *args):
        state_feats = None
        db_pointer = itertools.repeat([])
        graph_vec = itertools.repeat([])
        graph_ans_vec = itertools.repeat([])
        kb_entry_list = itertools.repeat([])
        x_tags = itertools.repeat([])
        utters, history_list, state_feats, db_pointer, graph_vec, graph_ans_vec, responses = args
        # if self.use_state_features:
        #     state_feats = args.pop()
        # if self.use_kb_attention:
        #     kb_entry_list = args.pop()
        # if self.use_db_features:
        #     db_pointer = args.pop()

        # responses = args.pop()
        # if self.use_ner_head:
        #     x_tags = args.pop()

        state_feats = state_feats or [[1]] * len(utters)

        if self.use_ner_head:
            assert all(len(u) == len(t) for u, t in zip(utters, x_tags)), \
                "utterance tokens and tags should have equal lengths"

        b_enc_ins, b_src_lens, b_dec_ins, b_dec_outs = [], [], [], []
        max_tgt_len = 0
        for x_tokens, hist_tok_list, y_tokens in zip(utters, history_list, responses):
            enc_in = self._encode_context(x_tokens, hist_tok_list)
            b_enc_ins.append(enc_in)
            b_src_lens.append(len(enc_in))

            dec_in, dec_out = self._encode_response(y_tokens)
            b_dec_ins.append(dec_in)
            b_dec_outs.append(dec_out)
            max_tgt_len = max(len(dec_in), max_tgt_len)

        # Sequence padding
        batch_size = len(b_enc_ins)
        max_src_len = max(b_src_lens)

        b_enc_ins_np = np.zeros((batch_size, max_src_len, self.embedding_size),
                                dtype=np.float32)
        b_src_tag_masks_np = np.zeros((batch_size, max_src_len), dtype=np.float32)
        b_src_tags_np = np.zeros((batch_size, max_src_len), dtype=np.float32)

        b_dec_ins_np = self.tgt_vocab[self.eos_token] *\
            np.ones((batch_size, max_tgt_len), dtype=np.float32)
        b_dec_outs_np = self.tgt_vocab[self.eos_token] *\
            np.ones((batch_size, max_tgt_len), dtype=np.float32)
        b_tgt_masks_np = np.zeros((batch_size, max_tgt_len), dtype=np.float32)
        b_kb_masks_np = np.zeros((batch_size, self.kb_size), dtype=np.float32)
        for i, (src_len, kb_entries, tags) in enumerate(zip(b_src_lens, kb_entry_list,
                                                            x_tags)):
            b_enc_ins_np[i, :src_len] = b_enc_ins[i]
            if len(tags):
                # TODO: debug examples with tags = [], no tokens in src sequence?
                b_src_tag_masks_np[i, src_len-len(tags):src_len] = 1.
                b_src_tags_np[i, src_len-len(tags):src_len] = tags
            tgt_len = len(b_dec_outs[i])
            b_dec_ins_np[i, :tgt_len] = b_dec_ins[i]
            b_dec_outs_np[i, :tgt_len] = b_dec_outs[i]
            b_tgt_masks_np[i, :tgt_len] = 1.

            if self.debug:
                if len(kb_entries) != len(set([e[0] for e in kb_entries])):
                    log.debug("Duplicates in kb_entries = {}".format(kb_entries))
            for k, v in kb_entries:
                b_kb_masks_np[i, self.kb_keys.index(k)] = 1.

        if self.use_ner_head:
            return (b_enc_ins_np, b_dec_ins_np, b_dec_outs_np, b_src_tags_np,
                    b_src_lens, b_tgt_masks_np, b_src_tag_masks_np,
                    state_feats, b_kb_masks_np)
        return (b_enc_ins_np, b_dec_ins_np, b_dec_outs_np,
                b_src_lens, b_tgt_masks_np, state_feats, b_kb_masks_np, db_pointer, graph_vec, graph_ans_vec)

    def train_on_batch(self, *args):
        return self.network.train_on_batch(*self.preprocess(*args))

    def _encode_context(self, tokens, hist_tok_list):
        hist_tokens = []
        for i in range(len(hist_tok_list) // 2):
            # history is in format [earlier turn .. later turn]
            hist_tokens += [self.delimiters[0]] + hist_tok_list[2*i]
            hist_tokens += [self.delimiters[1]] + hist_tok_list[2*i+1]
        tokens = hist_tokens + [self.delimiters[0]] + tokens
        if self.debug:
            log.debug("Context tokens = \"{}\"".format(tokens))
        return np.array(self.embedder([tokens])[0])

    def _encode_response(self, tokens):
        if self.debug:
            log.debug("Response tokens = \"{}\"".format(tokens))
        token_idxs = []
        for token in tokens:
            if token in self.kb_keys:
                token_idxs.append(self.tgt_vocab_size + self.kb_keys.index(token))
            else:
                token_idxs.append(self.tgt_vocab[token])
        # token_idxs = self.tgt_vocab([tokens])[0]
        return ([self.tgt_vocab[self.sos_token]] + token_idxs,
                token_idxs + [self.tgt_vocab[self.eos_token]])

    def _decode_response(self, token_idxs):
        def _idx2token(idxs):
            for idx in idxs:
                if idx < self.tgt_vocab_size:
                    token = self.tgt_vocab([[idx]])[0][0]
                    if token == self.eos_token:
                        break
                    yield token
                else:
                    yield self.kb_keys[idx - self.tgt_vocab_size]
        return [list(_idx2token(utter_idxs)) for utter_idxs in token_idxs]

    def __call__(self,
                 utters: List[List[str]],
                 history_list: List[List[List[str]]],
                 state_feats: List[List[Any]] = None,
                 db_pointer: List[Any] = None,
                 graph_vec: List[Any] = None,
                 graph_ans_vec: List[Any] = None,
                 kb_entry_list: List[dict] = itertools.repeat([])) ->\
            Tuple[List[str], List[float]]:
        b_enc_ins, b_src_lens = [], []
        if (len(utters) == 1) and not utters[0]:
            utters = [['hi']]
        for x_tokens, hist_tok_list in zip(utters, history_list):
            enc_in = self._encode_context(x_tokens, hist_tok_list)
            b_enc_ins.append(enc_in)
            b_src_lens.append(len(enc_in))
            # TODO: only last user utter is masked with ones, think of better way
        if state_feats is None:
            state_feats = [[1]] * len(b_enc_ins)

        # Sequence padding
        batch_size = len(b_enc_ins)
        max_src_len = max(b_src_lens)
        b_enc_ins_np = np.zeros((batch_size, max_src_len, self.embedding_size),
                                dtype=np.float32)
        b_src_tag_masks_np = np.zeros((batch_size, max_src_len), dtype=np.float32)
        b_kb_masks_np = np.zeros((batch_size, self.kb_size), dtype=np.float32)
        for i, (x_tokens, src_len, kb_entries) in enumerate(zip(utters, b_src_lens,
                                                                kb_entry_list)):
            b_enc_ins_np[i, :src_len] = b_enc_ins[i]
            b_src_tag_masks_np[i, src_len-len(x_tokens):src_len] = 1.
            if self.debug:
                log.debug("infer: kb_entries = {}".format(kb_entries))
            for k, v in kb_entries:
                b_kb_masks_np[i, self.kb_keys.index(k)] = 1.

        if self.use_ner_head:
            log.info(f"state features have shape {np.shape(state_feats)}")
            pred_idxs, tag_idxs = self.network(b_enc_ins_np, b_src_lens,
                                               b_src_tag_masks_np,
                                               state_feats, b_kb_masks_np)
            preds = self._decode_response(pred_idxs)
            return preds, [0.5] * len(preds), tag_idxs

        pred_idxs = self.network(b_enc_ins_np, b_src_lens, state_feats,
                                 b_kb_masks_np, db_pointer, graph_vec, graph_ans_vec)
        preds = self._decode_response(pred_idxs)
        if self.debug:
            log.debug("Dialog prediction = \"{}\"".format(preds[-1]))
        return preds, [0.5] * len(preds)

    def save(self):
        self.network.save()

    def load(self):
        pass

    def process_event(self, event_name, data):
        self.network.process_event(event_name, data)

    def destroy(self):
        self.embedder.destroy()
        self.network.destroy()
