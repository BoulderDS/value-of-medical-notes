import os
import pickle
import json
import argparse
from collections import Counter
import numpy as np
import re
import pandas as pd
import Constants
from nltk import word_tokenize

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.word_count = []

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_glove_voc(threshold, vocab, paragraph):
    data_path = '/corpus/glove/pretrained_vector/english/glove.42B.300d.{}'
    with open(data_path.format('json'),'r', encoding='utf-8') as f:
        glove = json.load(f, encoding='utf-8')

    count = 0
    word2vec = {}
    if paragraph:
        weight_matrix = np.random.uniform(-0.5, 0.5, size=(threshold+6,300))
    else:
        weight_matrix = np.random.uniform(-0.5, 0.5, size=(threshold+4,300))

    with open(data_path.format('txt'),'r', encoding='utf8') as f:
        for line in f:
            l = line.strip().split()
            word = l[0]
            if vocab(word) != 3:
                weight_matrix[vocab(word),:] = np.asarray(list(map(float, l[1:])))

            count += 1


    return weight_matrix

def clean_str_old(string):
    string = string.lower()
    string = re.sub(u"(\u2018|\u2019)", "'", string)
    string = re.sub(u'(\u201c|\u201d)', '"', string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\"s", " \'s", string)
    string = re.sub(r"\"ve", " \'ve", string)
    string = re.sub(r"n\"t", " n\'t", string)
    string = re.sub(r"\"re", " \'re", string)
    string = re.sub(r"\"d", " \'d", string)
    string = re.sub(r"\"ll", " \'ll", string)
    string = re.sub(r"\"m", " \'m", string)
    string = re.sub(r"\."," .", string)
    string = re.sub(r"\!"," !", string)
    string = re.sub(r"\,"," ,", string)
    #string = re.sub(r" "," ", string)
    return string

def clean_str(x):
    y=re.sub('\\[(.*?)\\]','',x) #remove de-identified brackets
    y=re.sub('[0-9]+\.','',y) #remove 1.2. since the segmenter segments based on this
    y=re.sub('dr\.','doctor',y)
    y=re.sub('m\.d\.','md',y)
    y=re.sub('admission date:','',y)
    y=re.sub('discharge date:','',y)
    y=re.sub('--|__|==','',y)
    return y


def build_vocab(list_files, dir_path, note_ids, threshold):
    """Build a simple vocabulary wrapper."""
    counter = Counter()

    df = pd.read_csv(list_files)
    print(len(df))
    for i, stay in enumerate(df['stay']):
        if i % 500 == 0 :
            print(i)
        for note_id in note_ids:
            note = pd.read_csv(os.path.join(dir_path, stay))
            note = " ".join([str(n) for n in note[note_id].dropna().values])
            note = note.lower()
            note = clean_str(note)
            tokens = word_tokenize(note)
            #tokens = [t for t in tokens]
            #print(tokens)
            counter.update(tokens)


    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    # Creates a vocab wrapper and add some special .
    vocab = Vocabulary()
    word_count = {}
    for word, cnt in counter.items():
        word_count[word] = cnt
    vocab.add_word(Constants.PAD_WORD)
    vocab.add_word(Constants.UNK_WORD)
    vocab.add_word(Constants.BOS_WORD)
    vocab.add_word(Constants.EOS_WORD)
    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    for word, idx in vocab.word2idx.items():
        if word in word_count.keys():
            count = word_count[word]
            vocab.word_count.append(1/count)
        else:
            vocab.word_count.append(int(1))
    return vocab

def main(args):
    if not os.path.exists(args.vocab_dir):
        os.makedirs(args.vocab_dir)
        print("Make Data Directory")
    vocab = build_vocab(args.list_file, args.data_path, args.note_ids,
                        threshold=args.threshold)
    #W = build_glove_voc(len(vocab), vocab, args.paragraph)
    vocab_path = os.path.join(args.vocab_dir, f'{args.note}_{args.period}_{args.task}_vocab.pkl')
    #weight_path = os.path.join(args.vocab_dir, 'W.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    #with open(weight_path, 'wb') as f:
    #    pickle.dump(W, f)

    print("Total vocabulary size: %d" %len(vocab))
    print(vocab.word2idx)
    print("Saved the vocabulary wrapper to '%s'" %vocab_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_dir', type=str, default='/data/joe/physician_notes/Deep-Average-Network',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--data_dir', type=str, default='/data/joe/physician_notes/Deep-Average-Network',
                        help='path for preprocessed mimic data')
    parser.add_argument('--threshold', type=int, default=10,
                        help='minimum word count threshold')
    parser.add_argument('--note', type=str, choices=['all', 'all_but_discharge', 'physician', 'physician_nursing', 'discharge'],
                        help='minimum word count threshold')
    parser.add_argument('--task', type=str, choices=['readmission', 'mortality'],
                        help='task type', )
    parser.add_argument('--period', type=str, choices=['24', '48', 'retro'],
                        help="note period", )
    args = parser.parse_args()
    args.data_path = f"{args.data_dir}/timeseries_features_{args.period}/note/"
    args.list_file = f"{args.data_dir}/{args.task}/{args.note}_note_train_{args.period}.csv"

    args.note_ids = Constants.note_type[args.note]

    main(args)

