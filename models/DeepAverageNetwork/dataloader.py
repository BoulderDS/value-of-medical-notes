import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from build_vocab import Vocabulary
import Constants
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import os,sys
from nltk import word_tokenize
sys.path.insert(0, '/home/joe/physician_notes/models/sentence_select/')
sys.path.insert(0, '/home/joe/physician_notes/')
import utils
import re


def clean_str(x):
    y = re.sub('\\[(.*?)\\]', '', x)  # remove de-identified brackets
    y = re.sub('[0-9]+\.', '', y)  # remove 1.2. since the segmenter segments based on this
    y = re.sub('dr\.', 'doctor', y)
    y = re.sub('m\.d\.', 'md', y)
    y = re.sub('admission date:', '', y)
    y = re.sub('discharge date:', '', y)
    y = re.sub('--|__|==', '', y)
    return y


class MIMICIIIDataset(Dataset):
    def __init__(
        self, vocab, listfiles, name, feature=None, 
        period="24", data_dir="", compare_note=None,
        text_length=None, segment=None
    ):
        self.listfiles = listfiles
        self.vocab = vocab
        self.max_len = 40000
        self.period = period
        self.stay2tokens = {}
        self.datapath = f'{data_dir}/timeseries_features_{period}/note/'
        self.note_ids = Constants.note_type[name]
        if compare_note:
            self.note_ids = [compare_note]
        self.compare_note = compare_note
        self.segment = segment
        self.features = feature
        self.text_length = text_length

    def __getitem__(self, index):
        stay = self.listfiles['stay'].iloc[index]
        mortality = self.listfiles['y_true'].iloc[index]
        if stay not in self.stay2tokens:
            df = pd.read_csv(os.path.join(self.datapath, stay))
            notes = []
            for note_id in self.note_ids:
                notes.extend([str(n) for n in df[note_id].dropna()])
                if self.compare_note and note_id in ["900001", "900005"]:
                    note_tmp_id = "900001" if note_id == "900005" else "900005"
                    notes.extend([str(n) for n in df[note_tmp_id].dropna()])
            # sentence select heuristics
            if self.segment:
                notes, _ = utils.segmentSentence(self, self.segment, notes, ["0"]*len(notes), self.datapath, stay, compare=True)

            notes = " ".join(notes).lower()
            notes = clean_str(notes)
            tokens = []
            tokenized_note = word_tokenize(notes)
            if self.text_length:
                tokens.extend([self.vocab(token) for i, token in enumerate(reversed(tokenized_note)) if i < self.text_length])
            else:
                tokens.extend([self.vocab(token) for i, token in enumerate(reversed(tokenized_note)) if i < self.max_len])
            self.stay2tokens[stay] = tokens
        else:
            tokens = self.stay2tokens[stay]

        if self.features is not None:
            feature = self.features[stay]
            return tokens, mortality, feature
        else:
            return tokens, mortality

    def collate_fn(self, data):
        # List of sentences and frames [B,]
        if self.features is not None:
            notes, mortality, features = zip(*data)
        else:
            notes, mortality = zip(*data)
        lengths = [len(x) if len(x) > 0 else 1 for x in notes]
        padded_notes = [n + [Constants.PAD for _ in range(self.max_len - len(n))] for n in notes]

        notes = torch.LongTensor(padded_notes).view(-1, self.max_len)
        mortality = torch.LongTensor(mortality).view(-1, 1)
        lengths = torch.LongTensor(lengths).view(-1,1)
        if self.features is not None:
            features = torch.FloatTensor(features).view(-1, len(features[0]))
            return notes, lengths, mortality, features
        else:
            return notes, lengths, mortality

    def __len__(self):
        return len(self.listfiles)


def impute_scale_features(train, val, test, period, data_dir):
    print("Loading features")
    features = pickle.load(open(f'{data_dir}/features_{period}.pkl','rb'))

    def load_data(listfile, features):
        stays, fs = [], []
        for stay in listfile['stay']:
            stays.append(stay)
            fs.append(features[stay])
        return stays, fs

    train_stays, train_fs = load_data(train, features)
    val_stays, val_fs = load_data(val, features)
    test_stays, test_fs = load_data(test, features)
    print("feature length:", len(train_fs[0]))
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import MinMaxScaler
    print("Impute features")
    if not os.path.exists(f'{data_dir}/features_imputer_{period}.pkl'):
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean.fit(train_fs)
        pickle.dump(imp_mean, open(f'{data_dir}/features_imputer_{period}.pkl', 'wb'))
    else:
        imp_mean = pickle.load(open(f'{data_dir}/features_imputer_{period}.pkl', 'rb'))
    train_fs = imp_mean.transform(train_fs)
    val_fs = imp_mean.transform(val_fs)
    test_fs = imp_mean.transform(test_fs)
    print("feature length:", len(train_fs[0]))
    print("Scale features")
    if not os.path.exists(f'{data_dir}/features_scaler_{period}.pkl'):
        scaler = MinMaxScaler()
        scaler.fit(train_fs)
        pickle.dump(scaler, open(f'{data_dir}/features_scaler_{period}.pkl', 'wb'))
    else:
        scaler = pickle.load(open(f'{data_dir}/features_scaler_{period}.pkl', 'rb'))
    train_fs = scaler.transform(train_fs)
    val_fs = scaler.transform(val_fs)
    test_fs = scaler.transform(test_fs)
    print("feature length:", len(train_fs[0]))
    return {key: value for (key, value) in zip(train_stays, train_fs)},\
           {key: value for (key, value) in zip(val_stays, val_fs)},\
           {key: value for (key, value) in zip(test_stays, test_fs)}, len(train_fs[0])


def get_loader(df, vocab, feature, name, batch_size,
        shuffle, num_workers, period, data_dir,
        compare_note, text_length, segment):
    
    Mimic = MIMICIIIDataset(vocab, df, name, feature, period, data_dir,
        compare_note=compare_note, text_length=text_length, segment=segment
    )

    data_loader = torch.utils.data.DataLoader(dataset=Mimic,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=Mimic.collate_fn)
    return data_loader


def get_loaders(args, is_test=False, is_feature=False):
    print(f"Note name : {args.name}")
    with open(f"{args.data_dir}/Deep-Average-Network/{args.name}_{args.period}_{args.task}_vocab.pkl",'rb') as f:
        print("----- Loading Vocab -----")
        vocab = pickle.load(f)
        print(f"vocab size: {len(vocab)}")
    print('----- Loading Note -----')

    TRAIN_NOTE_PATH = f"{args.data_dir}/{args.task}/{args.name}_note_train_{args.period}.csv"
    VAL_NOTE_PATH = f"{args.data_dir}/{args.task}/{args.name}_note_valid_{args.period}.csv"
    TEST_NOTE_PATH = f"{args.data_dir}/{args.task}/{args.name}_note_test_{args.period}.csv"

    train = pd.read_csv(TRAIN_NOTE_PATH)
    valid = pd.read_csv(VAL_NOTE_PATH)
    test = pd.read_csv(TEST_NOTE_PATH)

    print("train size", len(train))
    print("val size", len(valid))
    print("test size", len(test))
    print()
    print('----- Building Loaders -----')
    if is_feature:
        train_feature, val_feature, test_feature, feature_len = impute_scale_features(train, valid, test, args.period, args.data_dir)
    else:
        train_feature, val_feature, test_feature, feature_len = None, None, None, None

    train_loader = get_loader(train, vocab, train_feature, args.name, args.batch_size, True, 10, args.period, args.data_dir, args.compare_note, args.text_length, args.segment)
    valid_loader = get_loader(valid, vocab, val_feature, args.name, args.batch_size, True, 10, args.period, args.data_dir, args.compare_note, args.text_length, args.segment)
    test_loader = get_loader(test, vocab, test_feature, args.name, args.batch_size, False, 10, args.period, args.data_dir, args.compare_note, args.text_length, args.segment)
    return train_loader, valid_loader, test_loader, vocab, feature_len
