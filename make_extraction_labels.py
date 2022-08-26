"""Produce the dataset with (psudo) extraction label"""
import os
import json
import argparse
from time import time
from datetime import timedelta
import multiprocessing as mp
from cytoolz import curry, compose
from utils.utils import count_data
from utils.metric import compute_rouge_l


def _split_words(texts):
    return map(lambda t: t.split(), texts)


def get_extract_label(art_sents, abs_sents):
    """ greedily match summary sentences to article sentences"""
    extracted = []
    scores = []
    indices = list(range(len(art_sents)))
    for abst in abs_sents:
        rouges = list(map(compute_rouge_l(reference=abst, mode='r'), art_sents))
        ext = max(indices, key=lambda i: rouges[i])
        indices.remove(ext)
        extracted.append(ext)
        scores.append(rouges[ext])
        if not indices:
            break
    return extracted, scores

@curry
def process(data_dir, split, i):
    data_dir = os.path.join(data_dir, split)
    with open(os.path.join(data_dir, '{}.json'.format(i))) as f:
        data = json.loads(f.read())
    tokenize = compose(list, _split_words)
    art_sents = tokenize(data['article'])
    abs_sents = tokenize(data['abstract'])
    if art_sents and abs_sents: # some data contains empty article/abstract
        extracted, scores = get_extract_label(art_sents, abs_sents)
    else:
        extracted, scores = [], []
    data['extracted'] = extracted
    data['score'] = scores
    with open(os.path.join(data_dir, '{}.json'.format(i)), 'w') as f:
        json.dump(data, f, indent=4)


def label_mp(data_dir, split):
    """ process the data split with multi-processing"""
    start = time()
    print('start processing {} split...'.format(split))
    data_dir = os.path.join(data_dir, split)
    n_data = count_data(data_dir)
    with mp.Pool() as pool:
        list(pool.imap_unordered(process(data_dir, split),
                                 list(range(n_data)), chunksize=1024))
    print('finished in {}'.format(timedelta(seconds=time()-start)))


def label(data_dir, split):
    start = time()
    print('start processing {} split...'.format(split))
    data_dir = os.path.join(data_dir, split)
    n_data = count_data(data_dir)
    for i in range(n_data):
        print('processing {}/{} ({:.2f}%%)\r'.format(i, n_data, 100*i/n_data), end='')
        with open(os.path.join(data_dir, '{}.json'.format(i))) as f:
            data = json.loads(f.read())
        tokenize = compose(list, _split_words)
        art_sents = tokenize(data['article'])
        abs_sents = tokenize(data['abstract'])
        extracted, scores = get_extract_label(art_sents, abs_sents)
        data['extracted'] = extracted
        data['score'] = scores
        with open(os.path.join(data_dir, '{}.json'.format(i)), 'w') as f:
            json.dump(data, f, indent=4)
    print('finished in {}'.format(timedelta(seconds=time()-start)))


def main(args):
    for split in ['val', 'train']:  # no need of extraction label when testing
        label_mp(args.data, split)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train word2vec embedding used for model initialization'
    )
    parser.add_argument('--data', required=True, help='data directories')
    args = parser.parse_args()
    main(args)
