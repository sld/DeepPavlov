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

import pickle
from logging import getLogger
from typing import Iterator, Union
from itertools import chain
from pathlib import Path

from overrides import overrides
import numpy as np
from gensim.models import KeyedVectors
from gensim.utils import to_utf8

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.data.utils import flatten_str_batch 
from deeppavlov.models.embedders.abstract_embedder import Embedder
from deeppavlov.core.models.serializable import Serializable

log = getLogger(__name__)


@register('random_embedder')
class RandomEmbedder(Embedder):
    """
    Class implements random embeddings model

    Args:
        load_path: path where to load pre-trained embedding model from
        pad_zero: whether to pad samples or not
        dim: embedding size

    Attributes:
        model: random embeddings dictionary
        tok2emb: dictionary with already embedded tokens
        dim: dimension of embeddings
        pad_zero: whether to pad sequence of tokens with zeros or not
        load_path: path with pre-trained word2vec model
        save_path: path to save in word2vec format
    """
    def __init__(self,
                 load_path: Union[str, Path],
                 save_path: Union[str, Path] = None,
                 pad_zero: bool = False,
                 mean: bool = False,
                 dim: int = None,
                 **kwargs) -> None:
        """
        Initialize embedder with given parameters
        """
        Serializable.__init__(self, load_path=load_path, save_path=save_path)
        self.tok2emb = {}
        self.pad_zero = pad_zero
        self.mean = mean
        self.dim = dim
        self.model = None
        if self.load_path and self.load_path.exists():
            self.load()

    def fit(self, *args):
        if self.dim is None:
            raise ConfigError("dimension of random embeddings should be specified")
        self.model = {}
        tokens = set(chain(*flatten_str_batch(*args)))
        for tok in set(tokens):
            emb = np.random.randn(self.dim).astype(np.float32) / np.sqrt(self.dim)
            self.model[tok] = np.around(emb, 6)
        if self.save_path:
            self.save()

    def _get_word_vector(self, w: str) -> np.ndarray:
        return self.model[w]

    def load(self) -> None:
        """
        Load dict of embeddings from given file
        """
        log.info(f"[loading random embeddings from `{self.load_path}`]")
        if not self.load_path.exists():
            log.warning(f'{self.load_path} does not exist, cannot load embeddings from it!')
            return
        self.model = {}
        with open(self.load_path, 'rt') as fin:
            vocab_size, vector_size = map(int, fin.readline().split())
            for i in range(vocab_size):
                line = fin.readline()
                parts = line.rstrip().split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line {i}")
                self.model[parts[0]] = np.array(list(map(np.float32, parts[1:])))
        self.dim = vector_size

    def save(self):
        """
        Save random embeddings
        """
        log.info("[saving embeddings to {}]".format(self.save_path))

        total_vec = len(self.model)
        with open(self.save_path, 'wb') as fout:
            for i, word in enumerate(self.__iter__()):
                row = self.model[word]
                if i == 0:
                    vector_size = len(row)
                    fout.write(to_utf8("%s %s\n" % (total_vec, vector_size)))
                fout.write(to_utf8("%s %s\n" % (word, ' '.join("%.6f" % v for v in row))))

    @overrides
    def __iter__(self) -> Iterator[str]:
        """
        Iterate over all words from GloVe model vocabulary

        Returns:
            iterator
        """
        yield from self.model.keys()

    def serialize(self) -> bytes:
        return pickle.dumps(self.model, protocol=4)

    def deserialize(self, data: bytes) -> None:
        self.model = pickle.loads(data)
        self.dim = self.model.vector_size
