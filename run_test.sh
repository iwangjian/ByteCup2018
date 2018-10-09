CUDA_VISIBLE_DEVICES=5 python3 run_summarization.py --mode=decode --data_path=./data/tokenized/finished_files/chunked/test_* --vocab_path=./data/tokenized/finished_files/vocab --log_root=./log --exp_name=bytecup --single_pass=True
python3 commit_data.py
