import os
import json
import re
from summa import keywords
from scripts.retriever import Indexer, Queryer

TRAIN_DIR = "./data_20181124"
TEST_DIR = "./test"
INDEX_DIR = "./index"
VALID_DIR = "./valid"


def load_train(data_dir):
    ids_train = []
    contents_train = []
    titles_train = []
    count = 0
    for file in os.listdir(data_dir):
        if count > 0 and count % 10000 == 0:
            print("processed %d files" % count)
        with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f_obj:
            text_dict = json.load(f_obj)
            ids_train.append(str(text_dict["id"]))
            contents_train.append(str(text_dict["content"]))
            titles_train.append(str(text_dict["title"]))
        count += 1

    train_data = {"ids": ids_train,
                  "contents": contents_train,
                  "titles": titles_train}
    return train_data


def load_test(data_dir):
    ids_test = []
    contents_test = []

    for file in os.listdir(data_dir):
        with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f_obj:
            text_dict = json.load(f_obj)
            ids_test.append(text_dict["id"])
            content = " ".join(text_dict["content"])
            contents_test.append(content)
    test_data = {"ids": ids_test,
                 "contents": contents_test}
    return test_data


def build_mapping(data_train, index_dir):
    """
    Build mappings for train data: <id, content>, <id, title>
    :param data_train:
    :param index_dir:
    :return:
    """
    id2content = {}
    id2title = {}

    data_size = len(data_train["ids"])
    assert len(data_train["contents"]) == data_size
    assert len(data_train["titles"]) == data_size
    print("data_size:", data_size)

    for i in range(data_size):
        key = data_train["ids"][i]
        value_content = data_train["contents"][i]
        value_title = data_train["titles"][i]
        id2content[key] = value_content
        id2title[key] = value_title
    # write mappings to json files
    with open("%s/id2content.json" % index_dir, "w") as fw:
        json.dump(id2content, fw)
    with open("%s/id2title.json" % index_dir, "w") as fw:
        json.dump(id2title, fw)

    return id2content, id2title


def validate(query):
    query_words = query.split(" ")
    valid_query = []
    for w in query_words:
        if re.match(r"[A-Za-z]", w):
            valid_query.append(w)
    valid_query = str(" ".join(valid_query))
    return valid_query


def extract_query(text):
    top_words = keywords.keywords(text, split=True)
    query = " ".join(top_words)
    return query


def main():
    if not os.path.exists(INDEX_DIR):
        os.mkdir(INDEX_DIR)
    if not os.path.exists(VALID_DIR):
        os.mkdir(VALID_DIR)
    # load train data
    data_train = load_train(TRAIN_DIR)

    # build mappings for train data: <id, content>, <id, title>
    id2content, id2title = build_mapping(data_train, INDEX_DIR)

    # build indexes of train data
    indexer = Indexer(INDEX_DIR)
    indexer.build_index(id2content, id2title)

    # query top-k similar data of test set
    data_test = load_test(TEST_DIR)
    queryer = Queryer(INDEX_DIR, top_k=5)
    query_ids = data_test["ids"]
    query_contents = data_test["contents"]

    print("len querys:", len(query_ids))

    valid_ids = []
    for i in range(len(query_ids)):
        # retrieve by topic words
        #query = extract_query(query_contents[i])
        # retrieve by content
        query = validate(query_contents[i])

        print("query id:", query_ids[i])
        results = queryer.run_query(query)
        ids = results["ids"]
        titles = results["titles"]
        print("results:", ids)
        for title in titles:
            print("title:", title)
        for id in ids:
            valid_ids.append(id)

    # copy/move retrieved files to valid set
    print("copy files...")
    for id in valid_ids:
        cmd = "cp ./%s/%s.train.story ./%s/" % (TRAIN_DIR, id, VALID_DIR)
        os.system(cmd)
    valid_files = [name for name in os.listdir(VALID_DIR)]
    print("valid files:", len(valid_files))


if __name__ == "__main__":
    main()
