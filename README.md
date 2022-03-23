## Requirements
* PyTorch
* Transformers

## Training
The script for training is:
```
PYTHONENCODING=utf-8 python run_classifier.py --data_dir ./data/Ptacek \ 
--output_dir ./output/Ptacek_KL-Bert_output/ --do_train --do_test --model_select KL-Bert
```
where 
* `--data_dir` can be set as `./data/SARC_politics`, `./data/Ghosh`, and `./data/Ptacek`  
* `--model_select` can be set as `KL-Bert`, `Bert_concat`, and `Bert-Base`  
* `--output_dir` should keep up with `data_dir` and `model_select` to be `./output/DATASETNAME_MODELNAME_output/`
* `--know_strategy` is for different knowledge selecting strategies, which can be `common_know.txt`, `major_sent_know.txt`, and `minor_sent_know.txt`
* `--know_num` is to choose how many items of knowledge are used for each sentence, which is set to `'5'`, `'4'`, `'3'`, `'2'`, `'1'`
