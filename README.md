# Sentiment_Analysis_Imdb

## Introduction

We use the bert、roberta、sentiWSP totally 3 different models and using the gru、lstm、bilstm、textcnn、rnn、fnn totally 6 mothods to run on the imdb datasets. In the end, lstm+roberta was found to work best.

### Dataset

The dataset is sentiment binary classification, with 25k positive and 25k negative data. In addition, the training set has a total of 25k and the test set has a total of 25k.

### Expermental process

In addition to that, I've also covered the process of experimentation in detail on my blog, which you can take a look at
if you're interested Experimenttation
process  [CSDN_IMDB_Sentiment_Analysis](https://blog.csdn.net/ccaoshangfei/article/details/127537953?spm=1001.2014.3001.5501 )

### Network

The network structure is as follows

![Github版 IMDB](https://user-images.githubusercontent.com/105692522/198009720-8bfee092-1a10-41dd-9988-f51ef3ef89cb.png)

### Result




## Requirement

- Python = 3.9
- torch = 1.11.0
- numpy = 1.22.3

## Preparation

### Clone

```bash
git clone https://github.com/BeiCunNan/DataCastle_Sentiment_Analysis.git
```

### Create an anaconda environment

```bash
conda create -n sai python=3.9
conda activate sai
pip install -r requirements.txt
```

## Usage

```bash
python main.py --method sai
```
