import csv
import os
import re

import numpy as np
import pandas as pd


def get_data():
    raw_train_dataset = pd.read_csv("train.csv")
    raw_test_dataset = pd.read_csv("test_noLabel.csv")
    train_text = np.array(raw_train_dataset['txt'])
    train_label = np.array(raw_train_dataset['Label'])
    submit_text = np.array(raw_test_dataset['TXT'])
    submit_label= np.array([i for i in range(len(submit_text))])
    return train_text, train_label, submit_text,submit_label

def save_process():
    train_text, train_label, submit_text,submit_label = get_data()
    train_save_text,submit_save_text=[],[]
    punc = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]'
    # pre treat train_dataset
    for sen in train_text:
        sen = sen.replace('\n', '')
        sen = sen.replace('<br /><br />', ' ')
        sen = re.sub(punc, '', sen)
        train_save_text.append(sen)
    # pre treat test_dataset
    for sen in submit_text:
        sen = sen.replace('\n', '')
        sen = sen.replace('<br /><br />', ' ')
        sen = re.sub(punc, '', sen)
        submit_save_text.append(sen)
    # Save
    df_train = pd.DataFrame({'labels': train_label, 'sentences': train_save_text})
    df_train.to_csv("treat_train.csv", index=False)
    df_submit = pd.DataFrame({'labels': submit_label, 'sentences': submit_save_text})
    df_submit.to_csv("treat_submit.csv", index=False)

if __name__ == '__main__':
    save_process()
