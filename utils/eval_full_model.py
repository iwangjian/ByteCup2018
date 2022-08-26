""" Evaluate the baselines ont ROUGE/METEOR"""
import argparse
import json
import os
from evaluate import eval_meteor, eval_rouge


def main(args):
    dec_dir = os.path.join(args.decode_dir, 'output')
    with open(os.path.join(args.decode_dir, 'log.json')) as f:
        split = json.loads(f.read())['split']
    ref_dir = os.path.join(args.data, 'refs', split)
    assert os.path.exists(ref_dir)

    if args.rouge:
        dec_pattern = r'(\d+).dec'
        ref_pattern = '#ID#.ref'
        output = eval_rouge(dec_pattern, dec_dir, ref_pattern, ref_dir)
        metric = 'rouge'
    else:
        dec_pattern = '[0-9]+.dec'
        ref_pattern = '[0-9]+.ref'
        output = eval_meteor(dec_pattern, dec_dir, ref_pattern, ref_dir)
        metric = 'meteor'
    print(output)
    with open(os.path.join(args.decode_dir, '{}.txt'.format(metric)), 'w') as f:
        f.write(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate the output files for the RL full models')
    parser.add_argument('--data', required=True, help='data directories')
    parser.add_argument('--decode_dir', action='store', required=True,
                        help='directory of decoded summaries')

    # choose metric to evaluate
    metric_opt = parser.add_mutually_exclusive_group(required=True)
    metric_opt.add_argument('--rouge', action='store_true',
                            help='ROUGE evaluation')
    metric_opt.add_argument('--meteor', action='store_true',
                            help='METEOR evaluation')
    args = parser.parse_args()
    main(args)
