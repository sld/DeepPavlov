import pickle
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from nltk import word_tokenize
from tqdm import tqdm

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import download
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.estimator import Estimator

logger = get_logger(__name__)


@register('opensubtitles_preprocessor')
class OpenSubtitlesPreprocessor(Component):
    def __init__(self, *args, **kwargs):
        super()

    def __call__(self, source_sentences, **kwargs):
        res = []
        for source_sentence in source_sentences:
            source_tokens = [t for t in source_sentence.split(" ")]
            res.append(source_tokens)
        # target_tokens = [t for t in target_sentence.split(" ")]
        return res
