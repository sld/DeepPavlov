from pathlib import Path
import json
import random

from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress
from deeppavlov.core.common.registry import register


@register('opensubtitles_reader')
class OpensubtitlesReader(DatasetReader):
    def read(self, dir_path: str):
        data = []
        with open("{}/open_sub_ver3.5000".format(dir_path), 'r') as f:
            for line in f:
                src, tgt = line.split("\t")
                data.append((src, tgt))
        # random.shuffle(data)
        part80 = int(0.8*len(data))
        part90 = int(0.9*len(data))
        return {'train': data[:part80], 'valid': data[part80:part90], 'test': data[part90:]}
