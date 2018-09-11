"""
Data preprocessing code on raw dataset.
"""

import os
import json

DATA_DIR = './data/bytecup2018'

LISTS_DIR = './data/lists'
TRAIN_DIR = './data/train'
VALID_DIR = './data/valid'
TEST_DIR = './data/test'


def process_train_valid(file):
    ids = []
    contents = []
    titles = []

    with open(file, 'r', encoding='utf-8') as f_obj:
        texts = f_obj.readlines()

    for text in texts:
        text_dict = json.loads(text)
        ids.append(int(text_dict['id']))
        contents.append(text_dict['content'])
        titles.append(text_dict['title'])

    split = len(titles) - 100
    train_ids = ids[:split]
    valid_ids = ids[split:]
    train_contents = contents[:split]
    valid_contents = contents[split:]
    train_titles = titles[:split]
    valid_titles = titles[split:]
    train_text_idxs = []
    valid_text_idxs = []

    # Write train sets to files 'xxx.train.story'
    for idx in range(len(train_ids)):
        train_text_idx = str(train_ids[idx]) + '.train.story'
        train_text_idxs.append(train_text_idx)
        file_train = TRAIN_DIR + '/' + str(train_ids[idx]) + '.train.story'

        with open(file_train, 'w', encoding='utf-8') as file_obj:
            file_obj.write(train_contents[idx] + '\n')
            file_obj.write('\n')
            file_obj.write('@highlight' + '\n')
            file_obj.write('\n')
            file_obj.write(train_titles[idx] + '\n')
        #if int(idx) % 5000 == 0:
        #    print('Writing the %d file.' % idx)

    # Write validation sets to files 'xxx.valid.story'
    for idx in range(len(valid_ids)):
        valid_text_idx = str(valid_ids[idx]) + '.valid.story'
        valid_text_idxs.append(valid_text_idx)
        file_valid = VALID_DIR + '/' + str(valid_ids[idx]) + '.valid.story'

        with open(file_valid, 'w', encoding='utf-8') as file_obj:
            file_obj.write(valid_contents[idx] + '\n')
            file_obj.write('\n')
            file_obj.write('@highlight' + '\n')
            file_obj.write('\n')
            file_obj.write(valid_titles[idx] + '\n')
        #if int(idx) % 5000 == 0:
        #    print('Writing the %d file.' % idx)

    # Append indexes of train files
    train_file_idx = LISTS_DIR + '/all_train.txt'
    with open(train_file_idx, 'a', encoding='utf-8') as idx_obj:
        for idx_ in train_text_idxs:
            idx_obj.write(idx_ + '\n')

    # Append indexes of valid files
    valid_file_idx = LISTS_DIR + '/all_valid.txt'
    with open(valid_file_idx, 'a', encoding='utf-8') as idx_obj:
        for idx_ in valid_text_idxs:
            idx_obj.write(idx_ + '\n')


def process_test(file):
    ids = []
    contents = []
    test_text_idxs = []

    with open(file, 'r', encoding='utf-8') as f_obj:
        texts = f_obj.readlines()

    for text in texts:
        text_dict = json.loads(text)
        ids.append(int(text_dict['id']))
        contents.append(text_dict['content'])

    # Write test sets to files 'xxx.test.story'
    for idx in range(len(ids)):
        test_text_idx = str(ids[idx]) + '.test.story'
        test_text_idxs.append(test_text_idx)
        file_test = TEST_DIR + '/' + str(ids[idx]) + '.test.story'

        with open(file_test, 'w', encoding='utf-8') as file_obj:
            file_obj.write(contents[idx] + '\n')
            file_obj.write('\n')
           
        #if int(idx) % 5000 == 0:
        #    print('Writing the %d file.' % idx)
    # Append indexes of test files
    test_file_idx = LISTS_DIR + '/all_test.txt'
    with open(test_file_idx, 'w', encoding='utf-8') as idx_obj:
        for idx_ in test_text_idxs:
            idx_obj.write(idx_ + '\n')


if __name__ == '__main__':
    # Create data dirs if not exist
    DIRS = [LISTS_DIR, TRAIN_DIR, VALID_DIR, TEST_DIR]
    for dir in DIRS:
        if not os.path.exists(dir):
            os.mkdir(dir)

    # Process the training set
    for file in os.listdir(DATA_DIR):
        if 'bytecup.corpus.train' in str(file):
            print("Processing %s..." % file)
            file_path = os.path.join(DATA_DIR, file)
            process_train_valid(file_path)
            print("%s finished." % file)
        # Note: We temporarily use 'bytecup.corpus.validation_set' as test sets
        elif 'bytecup.corpus.validation_set' in str(file):
            print("Processing %s..." % file)
            file_path = os.path.join(DATA_DIR, file)
            process_test(file_path)
            print("%s finished." % file)