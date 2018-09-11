import os
import glob


decode_dir = glob.glob("./log/bytecup/decode_test_*/decoded/")[0]
result_dir = "./log/bytecup/result/"


def process_decoded(dec_dir, store_dir):
    for file in os.listdir(dec_dir):
        file_path = os.path.join(dec_dir, file)

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
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    process_decoded(decode_dir, result_dir)