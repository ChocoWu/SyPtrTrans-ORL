# SynPtrTrans-ORL
This repo is to implement the method described in the accapted paper [Mastering the Explicit Opinion-role Interaction: Syntax-aided Neural Transition System for Unified Opinion Role Labeling](https://www.aaai.org/AAAI22Papers/AAAI-729.WuS.pdf)

## Prerequisite
* [dynet Library](http://dynet.io/)
* [transformers](https://huggingface.co/transformers/model_doc/bert.html)
* [corenlp](https://stanfordnlp.github.io/CoreNLP/)
* [Glove](https://nlp.stanford.edu/projects/glove/)

## Usage (by examples)
### Data
Orignal data comes from [MPQA v2.0](https://mpqa.cs.pitt.edu/).


### Preprocessing
We need to obtain the dependency sturcture and POS tags for each data, and save as json format.
Pay attention to the file path and modify as needed

#### Get Dependency and POS
To parse the dependency structure and POS tags, we employ the [CoreNLP](https://stanfordnlp.github.io/CoreNLP/) provided by stanfordnlp.
So please download relavant files first.
```
  python preprocess.py
```

We also provide some preprocessed examples. 
If you want to use other datasets to train the model, please refer to the above steps.

### Train PtrTrans
```
  python train_ptrtrans.py
```
### Test
```
  python test.py
```

### Train SynPtrTrans
```
  python train_synptrtrans.py
```
### Test
```
  python test.py
```
