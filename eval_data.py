import os
import json
import numpy as np


def detect_length(dir, mode='train'):
    print("start computing length of %s data..." % mode)

    lens_article = []
    lens_abstract =[]
    files = os.listdir(dir)
    for file in files:
        f_path = os.path.join(dir, file)
        with open(f_path, 'r') as f_obj:
            lines = f_obj.readlines()
            art_sents = lines[0]
            if mode == 'test':
                abs_sents = ""
            else:
                abs_sents = lines[4]
            lens_article.append(len(art_sents.split()))
            lens_abstract.append(len(abs_sents.split()))

    print("articles:\nmax length={0}, min length={1}, avg_length={2}".format(
        np.max(lens_article), np.min(lens_article), np.average(lens_article)
    ))
    print("titles:\nmax length={0}, min length={1}, avg_length={2}".format(
        np.max(lens_abstract), np.min(lens_abstract), np.average(lens_abstract)
    ))


if __name__ == '__main__':
    TRAIN_DIR = './data/train'
    VALID_DIR = './data/valid'
    TEST_DIR = './data/test'

    detect_length(TEST_DIR, mode='test')
    detect_length(TRAIN_DIR, mode='train')
    detect_length(VALID_DIR, mode='valid')