import json


# FPATH = '/home/idris/.deeppavlov/downloads/multiwoz/multiwoz-for-deeppavlov.json'
FPATH = '/home/vimary/.deeppavlov/downloads/multiwoz/multiwoz-for-deeppavlov.json'
with open(FPATH, 'r') as f:
    data = json.load(f)


def get_one_hot(dialogue_name, raw_text, st):
    dialogue = data[dialogue_name]
    assert dialogue['thread'][st]['text'] == raw_text
    return dialogue['thread'][st]['one_hot_vec_for_cluster']
