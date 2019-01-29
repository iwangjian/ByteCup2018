# ByteCup2018

The topic of [Byte Cup 2018 International Machine Learning Contest](https://biendata.com/competition/bytecup2018/) is to  automatically generate titles of given articles. All data for training, validation and testing are from TopBuzz, a Bytedance's product, and other open sources.
  In this competition, we build a hybrid extractive-abstractive architecture with reinforcement learning (RL) based policy. The model first employs an extractor agent to select salient sentences or highlights, and then employs an abstractive network to rewrite the extracted sentences, using actor-critic policy gradient to learn the sentence saliency with dropout policy to avoid over-fitting.

## Dependencies
* Python3 (tested on Python 3.6)
* PyTorch 0.4
* gensim
* tensorboardX
* cytoolz
* pyrouge

## Quick Start
* Dataset

    We follow the instructions [here](https://github.com/ChenRocks/cnn-dailymail) for preprocessing the dataset. Meanwhile, we conduct data cleaning by removing duplicates (i.e., both content and title of 2 articles are the same) and cleaning some invalid characters (e.g., URLs, image comments, javascript strings, etc.). After that, all data files ```train```, ```val```, ```test``` and vocabulary file ```vocab_cnt.pkl``` are located in a specified data directory, e.g. ```./bytecup/finished_files/```.

* Pretrain word embeddings
```
python3 train_word2vec.py --data=./bytecup/finished_files --path=./bytecup/models/word2vec
```
* Make the pseudo-labels
```
python3 make_extraction_labels.py --data=./bytecup/finished_files
```
* Train abstractor and extractor
```
python3 train_abstractor.py --data=./bytecup/finished_files --path=./bytecup/models/abstractor --w2v=./bytecup/models/word2vec/word2vec.300d.332k.bin
python3 train_extractor.py --data=./bytecup/finished_files --path=./bytecup/models/extractor --w2v=./bytecup/models/word2vec/word2vec.300d.332k.bin
```
* Train the RL guided model
```
python3 train_full_rl.py --data=./bytecup/finished_files --path=./bytecup/models/save --abs_dir=./bytecup/models/abstractor --ext_dir=./bytecup/models/extractor
```
* Decode process
```
python3 decode_full_model.py --data=./bytecup/finished_files --path=./bytecup/output --model_dir=./bytecup/models/save --[val/test] 
```
* Convert decoded results for submission
```
python3 commit_data.py --decode_dir=./bytecup/output --result_dir=./bytecup/result
```

## References
[1] ["Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting"](http://aclweb.org/anthology/P18-1063) (ACL-18)

[2] ["Global Encoding for Abstractive Summarization"](http://aclweb.org/anthology/P18-2027) (ACL-18)

[3] ["Regularizing and Optimizing LSTM Language Models"](https://arxiv.org/pdf/1708.02182.pdf) (arXiv 2017)

[4] https://github.com/ChenRocks/fast_abs_rl
