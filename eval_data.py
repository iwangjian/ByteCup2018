import os
import json
import numpy as np


def detect_length(dir, mode='train'):
    print("start computing length of %s data..." % mode)

    idxs_file = []
    lens_article = []
    lens_abstract =[]
    files = os.listdir(dir)
    for file in files:
        f_path = os.path.join(dir, file)
        with open(f_path, 'r') as f_obj:
            sents = ""
            for line in f_obj.readlines():
                sents += line
            art_sents = sents.split('@highlight')[0].strip()
            if mode == 'test':
                abs_sents = ""
            else:
                abs_sents = sents.split('@highlight')[1].strip()
            idxs_file.append(file)
            lens_article.append(len(art_sents.split()))
            lens_abstract.append(len(abs_sents.split()))

    print("articles:\nmax length={0}, min length={1}, avg_length={2}".format(
        np.max(lens_article), np.min(lens_article), np.average(lens_article)
    ))
    idx_max = np.where(np.array(lens_article) == np.max(lens_article))
    idx_min = np.where(np.array(lens_article) == np.min(lens_article))
    for idx in idx_max[0]:
        print("max file:{0}".format(idxs_file[int(idx)]))
    for idx in idx_min[0]:
        print("min file:{0}".format(idxs_file[int(idx)]))

    if not mode == 'test':
        print("titles:\nmax length={0}, min length={1}, avg_length={2}".format(
            np.max(lens_abstract), np.min(lens_abstract), np.average(lens_abstract)
        ))
        idx_max = np.where(np.array(lens_abstract) == np.max(lens_abstract))
        idx_min = np.where(np.array(lens_abstract) == np.min(lens_abstract))
        for idx in idx_max[0]:
            print("max file:{0}".format(idxs_file[int(idx)]))
        for idx in idx_min[0]:
            print("min file:{0}".format(idxs_file[int(idx)]))


if __name__ == '__main__':
    TRAIN_DIR = './data/train'
    VALID_DIR = './data/valid'
    TEST_DIR = './data/test'

    detect_length(TEST_DIR, mode='test')
    print('\n')
    detect_length(TRAIN_DIR, mode='train')
    print('\n')
    detect_length(VALID_DIR, mode='valid')