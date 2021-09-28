import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np

def read_file(file_name):
    sents = []
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readline()
        for idx, l in enumerate(lines):
            if l[0] == ';' and lines[idx+1][0] == '$':
                this_sent = []
            elif l[0] == '$' and lines[idx - 1][0] ==';':
                continue
            elif l[0] == '/n':
                sents.append(this_sent)
            else:
                this_sent.append(tuple(l.split()))
    return sents

corpus = read_file(('train.txt'))

sentences, tags = [],[]
for t in corpus:
    tagged_sentence = []
    sentence, bio_tag = [],[]
    for w in t:
        tagged_sentence.append((w[1],w[3]))
        sentence.append(w[1])
        bio_tag.append(w[3])

    sentences.append(sentence)
    tags.append(bio_tag)

print("샘플 크기 : \n", len(sentences))
print("0번째 샘플 문장 시퀀스 : \n",sentences[0])
print("샘플 문장 시퀀스 최대 길이 : ",max(len[l] for l in sentences))
print("샘플 문장 시퀀스 평균 길이 : ", (sum(map(len,sentences))/len(sentences)))

sent_tokenizer = preprocessing.text.Tokenizer(oov_token='OOV')
sent_tokenizer.fit_on_texts(sentences)
tag_tokenizer = preprocessing.text.Tokenizer(lower=False)
tag_tokenizer.fit_on_texts(tags)

vocab_size = len(sent_tokenizer.word_index)+1
