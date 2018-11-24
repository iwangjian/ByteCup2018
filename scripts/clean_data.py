import os
import json

TRAIN_DIR = "./data_20181120"
CLEANED_TRAIN_DIR = "./data_20181124"


def clean_data():
    """Deduplicate data in TRAIN_DIR"""
    title2file = dict()
    title_set = set()
    for file in os.listdir(TRAIN_DIR):
        with open(os.path.join(TRAIN_DIR, file), "r") as f_obj:
            text_dict = json.load(f_obj)
            title = text_dict["title"]
            title_set.add(title)
            title2file[title] = file
    print("len unique titles:", len(title_set))

    if not os.path.exists(CLEANED_TRAIN_DIR):
        os.mkdir(CLEANED_TRAIN_DIR)
    for t in title_set:
        file = title2file[t]
        # copy/move files to CLEANED_TRAIN_DIR
        cmd = "cp %s/%s %s/" % (TRAIN_DIR, file, CLEANED_TRAIN_DIR)
        os.system(cmd)
    cleaned_train_files = [name for name in os.listdir(CLEANED_TRAIN_DIR)]
    print("cleaned train files:", len(cleaned_train_files))


if __name__ == "__main__":
    clean_data()
