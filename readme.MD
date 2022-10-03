# SynPtrTrans-ORL
This repo is to implement the method described in the accapted paper [Mastering the Explicit Opinion-role Interaction: Syntax-aided Neural Transition System for Unified Opinion Role Labeling](https://www.aaai.org/AAAI22Papers/AAAI-729.WuS.pdf)

## Introduction
Unified opinion role labeling (ORL) aims to detect all possible opinion structures of ‘opinion-holder-target’ in one shot, given a text. 
The existing transition-based unified method, unfortunately, is subject to longer opinion terms and fails to solve the term overlap issue. 
Current top performance has been achieved by employing the span-based graph model, which however still suffers from both high model complexity and insufficient interaction among opinions and roles. 
In this work, we investigate a novel solution by revisiting the transition architecture, and augmenting it with a pointer network (PointNet). 
The framework parses out all opinion structures in linear-time complexity, meanwhile breaks through the limitation of any length of terms with PointNet. 
To achieve the explicit opinion-role interactions, we further propose a unif ied dependency-opinion graph (UDOG), co-modeling the syntactic dependency structure and the partial opinion-role structure. 
We then devise a relation-centered graph aggregator (RCGA) to encode the multi-relational UDOG, where the resulting high-order representations are used to promote the predictions in the vanilla transition system.

<div align=center>
<img src="img/PtrTrans.pdf" width="75%" height="75%" />
</div>


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
