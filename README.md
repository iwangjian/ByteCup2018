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

Update later...

## References
[1] ["Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting"](http://aclweb.org/anthology/P18-1063) (ACL-18)

[2] ["Global Encoding for Abstractive Summarization"](http://aclweb.org/anthology/P18-2027) (ACL-18)

[3] ["Regularizing and Optimizing LSTM Language Models"](https://arxiv.org/pdf/1708.02182.pdf) (arXiv 2017)

[4] https://github.com/ChenRocks/fast_abs_rl
