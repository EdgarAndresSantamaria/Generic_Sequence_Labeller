## Sequence labelling Generic Engine

Based on the scripts [`run_ner.py`](https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py) 

Here we propose the generic tool `generic_sequence_labeller.py ` ready to use for any Sequence labelling task.

This framework supports high performance: CPU, GPU or GPU distributed learning.

This framework supports wide range of dataset format: [official datasets](https://huggingface.co/nlp/viewer/)  , (.json) based dataset or (.txt) based dataset. 

This framework is faster due to fast tokenization and the support of lightweight models.

This framework is stronger due to fully Transformers integration, easy to use configuration setup and generic data loaders from .txt, .json or official datasets.

This framework is dynamic due the abstract configuration layer for end user usage, any internal problem is automatically managed.

This framework is democratic due the possibility to use high performance pre-trained models.

This framework is deployable as module for bigger structures.

This framework is made for developers either researchers, notice some performance study in the below papper.


For any issue, idea or help pleaase contact eandres011@ikasle.ehu.eus

Best regards Edgar Andrés

#### Enviroment settings

All the provided commands are prepared for Ubuntu 16.04 LTS > , also available for Windows pycharm .

The framework is prepared for python 3.6 > environment with the following dependencies:

1. transformers 3.5.0 > + tokenizers 0.9.3 >
2. torch 1.4.0
2. tensorflow 2.1.0

All dependencies are supported into maintained libraries that could be installed via:

1. `pip install transformers==3.5`
2. `pip install torch==1.4.0`
2. `pip install tensorflow==2.1.0`

For technical details visit the following pages:

1. [huggingface.co](https://huggingface.co/transformers/)
2. [pytorch.org](https://pytorch.org/)
3. [pytorch-crf Docs](https://pytorch-crf.readthedocs.io/en/stable/)
4. [tensorflow Docs](https://www.tensorflow.org/api_docs)


#### Sequence labelling data formats 

We can feed the network with multiple data sources we have three main inputs: official datasets, (.txt) files and (.json) files.
Those official datasets must fullfill the provided (.json) format in order to apply token sequence classification task as for example 'conll2003'.
If we decide to use our own data we must generate train, dev and test files. Those must fill (.txt) xor (.json) format.

Files (.txt) format:

```conll
Masaje B-Action
suave B-Concept
sobre O
el O
musculo B-Concept
. O

Ellos O
pueden O
aliviar B-Action
el O
dolor B-Concept
. O

...
```
For test.txt could be provided with or without label column.
The framework internally converts the provided (.txt) formats into compatible (.json) formats, this allow supporting multiple formats.

Files (.json) format:

```json

{"data": 
[
{"tokens": ["Masaje", "suave", "sobre", "el", "musculo", "."], "labels": ["B-Action", "B-Concept", "O", "O", "B-Concept", "O"]},
"..." 
]
}

```

#### JSON-based configuration file for experiments (under development: document all parameters)

Instead of passing all parameters via commandline arguments, the `run.py` script also supports reading parameters from a json-based configuration file as the following:

```json
{
    "model_name_or_path": "str = Path to pretrained model or model identifier from huggingface.co/models",
    "config_name": "Optional[str] = Pretrained config name or path if not the same as model_name",
    "tokenizer_name": "Optional[str] = Pretrained tokenizer name or path if not the same as model_name",
    "use_fast": "bool = Set this flag to use fast tokenization.",
    "cache_dir": "Optional[str] = Where do you want to store the pretrained models downloaded from s3",
  

    "dataset_txt": "bool = Indicates that entrance end with .txt",
    "dataset_json": "bool = Indicates that entrance end with .json",
    "train_file": "path to train file , must fullfill the format specified",
    "validation_file": "path to dev file , must fullfill the format specified",
    "test_file": "path to test file , must fullfill the format specified",
    "dataset_name": "automatically loads the desired dataset (without further config required) check -> https://huggingface.co/nlp/viewer/",
  
    "max_seq_length": "int = The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.",
    "overwrite_cache": "bool = Overwrite the cached training and evaluation sets",
    "num_train_epochs": "int = The maximum total Epochs to train.",
    "per_device_train_batch_size":  "int = The maximum total distributed batch size for train.",
    "per_device_eval_batch_size":  "int = The maximum total distributed batch size for eval or test.",
    "save_steps": "int = The partial steps until checkpoint save.",
    "seed": "int = The replicability seed.",
    "output_dir": "str = The output data dir. ",
    "do_train":"bool = if we want to train",
    "do_eval": "bool = if we want to evaluate",
    "do_predict": "bool = if we want to predict",
    "load_best_model_at_end" : "bool = checks the best checkpoint saved and retrieves it (specifies the epoch)",
    "metric_for_best_model": "str = metric to compare evaluation (loss by default) :'loss','f1','accuracy_score','precision' or 'racall'",
    "evaluation_strategy":"str = strategy to evaluate the mode (no by default): 'no' or 'steps' ",
    "eval_steps": "int = Steps until perform evaluation (only if 'evaluation_strategy' = 'steps')."
}
```
For example, we could compose the following experiment:

```json
{
    "dataset_txt": true,
    "train_file": "path_to_train/train.txt",
    "validation_file": "path_to_dev/dev.txt",
    "test_file": "path_to_test/test.txt",
    "model_name_or_path": "xlm-roberta-base",
    "output_dir": "path_to_out/",
    "load_best_model_at_end": true,
    "metric_for_best_model": "f1",
    "evaluation_strategy":"steps",
    "overwrite_output_dir": true,
    "max_seq_length": 13,
    "num_train_epochs": 4,
    "per_device_train_batch_size": 5,
    "per_device_eval_batch_size": 5,
    "eval_steps": 400,
    "seed": 1,
    "do_train": true,
    "do_eval": true,
    "do_predict": true
}
```

The different configuration result on distinct experiments, the program will display default verbose of the process.
It must be saved with a `.json` extension and can be used by running `python3.6 run.py config.json`.

#### Training

At every time the framework by default logs the available information of: the system, data and training status.

The train of a new system outcomes into output_dir 

1) The binaries of: the model, the training arguments and the sentencepiece bpe.

2) The configuration of : the model and tokenizer. 

3) Checkpoints per steps if enabled the functionality.


#### Evaluation

Evaluation on Sequence labelling outputs the following information into terminal:

```bash
I1124 18:07:47.169603 140330645186304 generic_sequence_labeller.py:428]   eval_loss = x.x
I1124 18:07:47.169718 140330645186304 generic_sequence_labeller.py:428]   eval_accuracy_score = x.x
I1124 18:07:47.169780 140330645186304 generic_sequence_labeller.py:428]   eval_precision = x.x
I1124 18:07:47.169835 140330645186304 generic_sequence_labeller.py:428]   eval_recall = x.x
I1124 18:07:47.169888 140330645186304 generic_sequence_labeller.py:428]   eval_f1 = x.x
```

The information is saved in the eval_results_ner.txt file:

```bash
eval_loss = x.x
eval_accuracy_score = x.x
eval_precision = x.x
eval_recall = x.x
eval_f1 = x.x
epoch = x.x
```

#### Prediction

The predictions will be returned as test_predictions.txt alongside the evaluation test_results.txt with the following formats:

test_predictions.txt
```conll
Masaje B-Action
suave B-Concept
sobre O
el O
musculo B-Concept
. O

Ellos O
pueden O
aliviar B-Action
el O
dolor B-Concept
. O

...
```

test_results.txt

```bash
eval_loss = x.x
eval_accuracy_score = x.x
eval_precision = x.x
eval_recall = x.x
eval_f1 = x.x

```

#### Quick start experiments

As we said before, all the experiments are easy driven by (.json) configuration files.

Here we asumme that well formated: train.txt, dev.txt and test.txt; are feed into the framework.

Never mind about the codification origins for the data, the only restriction is that all characters are convertible into utf-8 standard.

Notice that all the options are interchangeable and well designed to interact in the pipeline.

##### Train a system returning best evaluation f1

```json
{
    "dataset_txt": true,
    "train_file": "path_to/train.txt",
    "validation_file": "path_to/dev.txt",
    "model_name_or_path": "desired_model_identifier",
    "output_dir": "path_to/out/",
    "load_best_model_at_end": true,
    "metric_for_best_model": "f1",
    "evaluation_strategy":"steps",
    "max_seq_length": 128,
    "num_train_epochs": 5,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "eval_steps": 400,
    "seed": 1,
    "do_train": true,
    "do_eval": true,
    "do_predict": false
}
```

##### Train a model regardless evaluation
```json
{
    "dataset_txt": true,
    "train_file": "path_to/train.txt",
    "validation_file": "path_to/dev.txt",
    "model_name_or_path": "desired_model_identifier",
    "output_dir": "path_to/out/",
    "max_seq_length": 128,
    "num_train_epochs": 5,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "seed": 1,
    "do_train": true,
    "do_eval": true,
    "do_predict": false
}
```

##### predict with trained model
```json
{
    "dataset_txt": true,
    "test_file": "path_to/test.txt",
    "model_name_or_path": "path_to_trained_model/",
    "output_dir": "path_to/out/",
    "max_seq_length": 128,
    "num_train_epochs": 5,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "seed": 1,
    "do_train": false,
    "do_eval": false,
    "do_predict": true
}
```
##### complete pipeline

```json
{
    "dataset_txt": true,
    "train_file": "path_to/train.txt",
    "validation_file": "path_to/dev.txt",
    "test_file": "path_to/test.txt",
    "model_name_or_path": "desired_model_identifier",
    "output_dir": "path_to/out/",
    "load_best_model_at_end": true,
    "metric_for_best_model": "f1",
    "evaluation_strategy":"steps",
    "max_seq_length": 128,
    "num_train_epochs": 5,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "eval_steps": 400,
    "seed": 1,
    "do_train": true,
    "do_eval": true,
    "do_predict": true
}
```
## Citation
We first time presented the framework on [paper](https://addi.ehu.eus/handle/10810/48623):
```bibtex
@article{santamaria2020end,
  title={End to end approach for i2b2 2012 challenge based on Cross-lingual models},
  author={Santamar{\'\i}a, Edgar Andr{\'e}s},
  year={2020}
}
```