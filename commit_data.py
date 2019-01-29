import os
import argparse


def process_decoded(args):
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    punct = ["/", "`", "+", "-", ";", "-lrb-", "-rrb-", "``", "|", "~", "&quot"]
    for file in os.listdir(args.decode_dir):
        file_path = os.path.join(args.decode_dir, file)

        file_id = int(str(file).split('.')[0]) + 1

        res_file = str(file_id) + '.txt'
        res_path = os.path.join(args.result_dir, res_file)
        temp = []
        with open(file_path, 'r') as fr:
            text = fr.read().strip()
            data = text.split(" ")
            for word in data:
                if not word in punct:
                    temp.append(word)
            with open(res_path, 'w', encoding='utf-8') as fw:
                fw.write(" ".join(temp))
                fw.write('\n')
    print("Finished: %s" % args.result_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert decoded files to commit files')
    parser.add_argument('--decode_dir', action='store', required=True,
                        help='directory of decoded summaries')
    parser.add_argument('--result_dir', action='store', required=True,
                        help='directory of submission')
    args = parser.parse_args()

    process_decoded(args)
