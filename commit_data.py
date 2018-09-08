import os
from nltk.parse.corenlp import CoreNLPParser


PUNKS = [',', '.', '\'', ':', '!']


def process_decoded(dec_dir, store_dir):
    for file in os.listdir(dec_dir):
        file_path = os.path.join(dec_dir, file)
        '''
        with open(file_path, 'r') as f_obj:
            text = f_obj.readlines()
            #stanford = CoreNLPParser()
            #token = list(stanford.tokenize(text))
            token = []
            for line in text:
                words = line.split()
                for w in words:
                    token.append(w)
            token_res = ""
            for w in token:
                if w in PUNKS:
                    token_res += w
                else:
                    token_res += " "
                    token_res += w
        '''
        file_id = int(str(file).split('_')[0]) + 1
        print("file id: ", file_id)
        res_file = str(file_id) + '.txt'
        res_path = os.path.join(store_dir, res_file)

        with open(file_path, 'r') as fr:
            text = fr.read()
            with open(res_path, 'w', encoding='utf-8') as fw:
                fw.write(text)
                fw.write('\n')


if __name__ == '__main__':
    dec_dir = './log/bytecup/decode_test_300maxenc_4beam_4mindec_30maxdec_ckpt-390702/decoded/'
    store_dir = './log/bytecup/result/'
    if not os.path.exists(store_dir):
        os.mkdir(store_dir)

    process_decoded(dec_dir, store_dir)