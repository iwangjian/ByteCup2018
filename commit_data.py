import os
import util
import tensorflow as tf


result_dir = "./log/bytecup/result/"
FLAGS = tf.app.flags.FLAGS


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


def get_decode_dir_name(ckpt_name):
    if "train" in FLAGS.data_path:
        dataset = "train"
    elif "val" in FLAGS.data_path:
        dataset = "val"
    elif "test" in FLAGS.data_path:
        dataset = "test"
    else:
        raise ValueError("FLAGS.data_path %s should contain one of train, val or test" % (FLAGS.data_path))
    dirname = "decode_%s_%imaxenc_%ibeam_%imindec_%imaxdec" % \
              (dataset, FLAGS.max_enc_steps, FLAGS.beam_size, FLAGS.min_dec_steps, FLAGS.max_dec_steps)
    if ckpt_name is not None:
        dirname += "_%s" % ckpt_name
    return dirname


if __name__ == '__main__':
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # Load an initial checkpoint to use for decoding
    ckpt_path = util.load_ckpt(tf.train.Saver(), tf.Session(config=util.get_config()))
    # this is something of the form "ckpt-123456"
    ckpt_name = "ckpt-" + ckpt_path.split('-')[-1]
    decode_dir = os.path.join(FLAGS.log_root, get_decode_dir_name(ckpt_name))

    process_decoded(decode_dir, result_dir)