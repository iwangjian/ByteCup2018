""" make reference text files needed for ROUGE evaluation """
import json
import os
import argparse
from time import time
from datetime import timedelta
from utils import count_data
from decoding import make_html_safe


def dump(data_dir, split):
    start = time()
    print('start processing {} split...'.format(split))
    data_dir = os.path.join(data_dir, split)
    dump_dir = os.path.join(data_dir, 'refs', split)
    n_data = count_data(data_dir)
    for i in range(n_data):
        print('processing {}/{} ({:.2f}%%)\r'.format(i, n_data, 100*i/n_data), end='')
        with open(os.path.join(data_dir, '{}.json'.format(i))) as f:
            data = json.loads(f.read())
        abs_sents = data['abstract']
        with open(os.path.join(dump_dir, '{}.ref'.format(i)), 'w') as f:
            f.write(make_html_safe('\n'.join(abs_sents)))
    print('finished in {}'.format(timedelta(seconds=time()-start)))


def main(args):
    data_dir = args.data
    for split in ['val', 'test']:  # evaluation of train data takes too long
        if not os.path.exists(os.path.join(data_dir, 'refs', split)):
            os.makedirs(os.path.join(data_dir, 'refs', split))
        dump(data_dir, split)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make evaluations of the full model (RL)')
    parser.add_argument('--data', required=True, help='data directories')
    args = parser.parse_args()
    main(args)
