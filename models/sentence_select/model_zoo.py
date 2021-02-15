import models.config as Config
from in_hospital_mortality.custom_metrics import mortality_rate_at_k, train_val_compute
from in_hospital_mortality.feature_definitions import BOWFeatures, DictFeatures
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler #, StandardScalar
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle
import pandas as pd
import models.sentence_select.utils as utils
from models.sentence_select.utils import Vocabulary
from models.DeepAverageNetwork.model import *
import os
import numpy as np
import spacy
from nltk import word_tokenize
import torch
from tqdm import tqdm
import re
from nltk import sent_tokenize


nlp = spacy.load('en_core_web_sm', disable=['tokenizer', 'ner', 'textcat'])
class LR():
    def __init__(self, args):
        model_name = args.note+"_"+args.feature_period + '.chkpt'
        if args.feature_used == "all":
            model_name = "feature_text_" + model_name
        elif args.feature_used == "all_but_notes":
            model_name = "feature_" + model_name
        else:
            model_name = "text_" + model_name
        self.args = args
        self.model_name = model_name
        path = f'{args.data}/logistic_regression/models/{args.task}/{model_name}'
        self.model = pickle.load(open(path, 'rb'))
        self.nlp = spacy.load('en_core_web_sm', disable=['tokenizer', 'ner', 'textcat'])

    def predict(self):
        note, stays, y_label, from_notes = self.getData() # segmented note
        print("Run prediction")
        probs = self.model.predict_proba(note)[:,1]
        utils.findHighestScore(probs, note['file_name'], note['text'], y_label, from_notes, self.args)


    def getData(self):
        test_list = pd.read_csv(f"{self.args.data}/{self.args.task}/{self.args.note}_note_{self.args.split}_{self.args.feature_period}.csv")
        print("Loading data")
        result_path = self.model_name[:-6]+".csv"
        result_path = result_path.replace("feature_", "")
        if self.args.filter and self.args.note == "all_but_discharge":
            result_path = self.args.filter + "_" + result_path
        print(result_path)
        if os.path.exists(f"{self.args.data}/select_sentence/{self.args.model}/{self.args.task}/{self.args.segment}/{result_path}"):
        #if os.path.exists(f"/data/joe/pasdfhas"):
            test_list = pd.read_csv(f"{self.args.data}/select_sentence/{self.args.model}/{self.args.task}/{self.args.segment}/{result_path}").fillna("")
            #print(test_list)
            stays, notes, from_notes, y_test = test_list["stay"].values, test_list['bestSents'].values, test_list["note_id"].values, test_list['y_label'].values
            test_notes = pd.DataFrame({'file_name': stays, 'text': notes})
        else:
            X_test_notes, y_test, stays, from_notes = [], [], [], []
            note_ids = Config.note_type[self.args.note]
            datapath = os.path.join(self.args.data, f"timeseries_features_{self.args.feature_period}")
            print("test admision number:", len(test_list))
            for index, row in test_list.iterrows():
                note = pd.read_csv(os.path.join(datapath, "note", row['stay']), dtype=object).fillna("")
                note_collect = []
                tmp_from_notes = []

                for note_id in note_ids:
                    notes = [str(v) for v in note[note_id] if str(v) != '']
                    if notes:
                        note_collect.extend(notes)
                        tmp_from_notes.extend([note_id]*len(notes))

                sentences, tmp_from_notes = utils.segmentSentence(self, self.args.segment, note_collect, tmp_from_notes, datapath, row['stay'])
                X_test_notes.extend(sentences)
                if self.args.reverse_label:
                    y_test.extend([1 - row['y_true']]*len(sentences))
                else:
                    y_test.extend([row['y_true']]*len(sentences))

                from_notes.extend(tmp_from_notes)
                stays.extend([row['stay'] for _ in range(len(sentences))])
            test_notes = pd.DataFrame({'file_name': stays, 'text': X_test_notes})
        #print(test_notes)
        return test_notes, test_list['stay'], y_test, from_notes


class DeepAverageNetwork:
    def __init__(self, opt):
        self.args = opt
        with open(f"{self.args.data}/Deep-Average-Network/{opt.note}_{opt.feature_period}_{opt.task}_vocab.pkl",'rb') as f:
            print("----- Loading Vocab -----")
            self.vocab = pickle.load(f)
            print(f"vocab size: {len(self.vocab)}")
        self.device = torch.device(f'cuda:{opt.device}')
        if self.args.feature_used == "all":
            result_path = "text_" + self.args.note+"_"+self.args.feature_period + '.csv'
            if self.args.filter  and self.args.note == "all_but_discharge":
                result_path = self.args.filter + "_" + result_path
            test_list = pd.read_csv(f"{self.args.data}/select_sentence/{self.args.model}/{self.args.task}/{self.args.segment}/{result_path}").fillna("")
            self.features, self.feature_len = utils.impute_scale_features(
                                                test_list, 
                                                self.args.feature_period)
        else:
            self.feature_len = 1000
        self.dan = DAN(len(self.vocab), 300, self.feature_len , 0.5, opt.feature_used=="all", True)
        model_name = opt.task+'_'+opt.note +'_'+ opt.feature_period + '.chkpt'
        if opt.feature_used == "all":
            model_name = "feature_text_" + model_name
        elif opt.feature_used == "all_but_notes":
            model_name = "feature_" + model_name
        else:
            model_name = "text_" + model_name

        self.model_name = model_name
        self.args = opt
        checkpoint = torch.load(f"{self.args.data}/Deep-Average-Network/models/{model_name}", map_location='cpu')
        self.dan.load_state_dict(checkpoint['model'])
        self.dan.eval()
        self.max_len = 4000
        # load tfitest_list vectorizer
        model_name = opt.note+"_"+opt.feature_period + '.chkpt'
        if opt.feature_used == "all":
            model_name = "feature_text_" + model_name
        elif opt.feature_used == "all_but_notes":
            model_name = "feature_" + model_name
        else:
            model_name = "text_" + model_name
        path = f'{self.args.data}/logistic_regression/models/{opt.task}/{model_name}'
        self.model = pickle.load(open(path, 'rb'))
    
    def clean_str(self, x):
        y=re.sub('\\[(.*?)\\]','',x) #remove de-identified brackets
        y=re.sub('[0-9]+\.','',y) #remove 1.2. since the segmenter segments based on this
        y=re.sub('dr\.','doctor',y)
        y=re.sub('m\.d\.','md',y)
        y=re.sub('admission date:','',y)
        y=re.sub('discharge date:','',y)
        y=re.sub('--|__|==','',y)
        return y
    def predict(self):
        notes, tokens, stays, y_labels, from_notes = self.getData() # segmented note
        dataloader = self.buildDataLoader(tokens, y_labels)
        print("Run prediction")
        probs = self.run(dataloader)
        utils.findHighestScore(probs, stays, notes, y_labels, from_notes, self.args)


    def getData(self):
        test_list = pd.read_csv(f"{self.args.data}/{self.args.task}/{self.args.note}_note_{self.args.split}_{self.args.feature_period}.csv")
        print("Loading data")
        result_path = "text_" + self.args.note+"_"+self.args.feature_period + '.csv'
        if self.args.filter  and self.args.note == "all_but_discharge":
            result_path = self.args.filter + "_" + result_path
        #if os.path.exists(f"/data/joe/physician_notes/select_sentence/{self.args.model}/{self.args.task}/{self.args.segment}/{result_path}"):
        if os.path.exists(f"{self.args.data}/select_sentence/{self.args.model}/{self.args.task}/{self.args.segment}/{result_path}"):
            test_list = pd.read_csv(f"{self.args.data}/select_sentence/{self.args.model}/{self.args.task}/{self.args.segment}/{result_path}").fillna("")
            #print(test_list)
            stays, X_test_notes, from_notes, y_test = test_list["stay"].values, test_list['bestSents'].values, test_list["note_id"].values, test_list['y_label'].values

        else:
            X_test_notes, y_test, from_notes, stays = [], [], [], []
            note_ids = Config.note_type[self.args.note]
            datapath = os.path.join(self.args.data, f"timeseries_features_{self.args.feature_period}")
            print("test admision number:", len(test_list))
            for index, row in test_list.iterrows():
                note = pd.read_csv(os.path.join(datapath, "note", row['stay']), dtype=object).fillna("")
                note_collect = []
                tmp_from_notes = []

                for note_id in note_ids:
                    notes = [str(v) for v in note[note_id] if str(v) != '']
                    if notes:
                        note_collect.extend(notes)
                        tmp_from_notes.extend([note_id]*len(notes))

                sentences, tmp_from_notes = utils.segmentSentence(self, self.args.segment, note_collect, tmp_from_notes, datapath, row['stay'])
                X_test_notes.extend(sentences)
                if self.args.reverse_label:
                    y_test.extend([1 - row['y_true']]*len(sentences))
                else:
                    y_test.extend([row['y_true']]*len(sentences))

                from_notes.extend(tmp_from_notes)
                stays.extend([row['stay'] for _ in range(len(sentences))])
        # convert words to tokens
        test_notes = []
        for sent in X_test_notes:
            sent = self.clean_str(sent.lower())
            tokenized_sent = word_tokenize(sent)
            test_notes.append([self.vocab(token) for i, token in enumerate(reversed(tokenized_sent)) if i<self.max_len])
        return X_test_notes, test_notes, stays, y_test, from_notes

    def buildDataLoader(self, notes, labels):
        data_size = len(notes)
        batch_size = 256
        n_batch = data_size//batch_size + 1
        for i in range(n_batch):
            batch_notes = notes[i*batch_size:(i+1)*batch_size]
            batch_labels = labels[i*batch_size:(i+1)*batch_size]
            if self.args.feature_used == "all":
                batch_features = torch.FloatTensor(self.features[i*batch_size:(i+1)*batch_size]).view(-1, self.feature_len)
            #B = len(batch_notes)

            lengths = [len(x) if len(x) > 0 else 1 for x in batch_notes]
            padded_notes = [n + [0 for _ in range(self.max_len - len(n))] for n in batch_notes] #pad with 0 = Constant.PAD

            batch_notes = torch.LongTensor(padded_notes).view(-1, self.max_len)
            batch_labels = torch.LongTensor(batch_labels).view(-1, 1)
            lengths = torch.LongTensor(lengths).view(-1,1)
            if self.args.feature_used == "all":
                yield batch_notes, batch_labels, lengths, batch_features
            else:
                yield batch_notes, batch_labels, lengths, lengths
    def encode(self, sentences):
        # encode list of sentences to list of mean embeddings
        self.dan.to(self.device)
        self.dan.eval()
        tokens = []
        for sent in sentences:
            sent = self.clean_str(sent.lower())
            tokenized_sent = word_tokenize(sent)
            tokens.append([self.vocab(token) for i, token in enumerate(reversed(tokenized_sent)) if i<self.max_len])
        pesudo_labels = np.ones(shape=(len(tokens),))
        dataloader = self.buildDataLoader(tokens, pesudo_labels)
        output = []
        with torch.no_grad():
            for batch in dataloader:
                notes, labels, lengths = map(lambda x: x.to(self.device), batch)
                embeds = self.dan.embed(notes)
                embeds = torch.sum(embeds, dim=1)
                embeds /= lengths.float()
                #embeds = self.dan.batchnorm_text(embeds)
                if len(embeds.size()) == 1:
                    embeds = embeds.unsqueeze(0)
                #print(embeds.size())
                output.append(embeds.cpu().numpy())
        output = np.concatenate(output, axis=0)
        return output

    def run(self, dataloader):
        self.dan.to(self.device)
        self.dan.eval()
        probs = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                notes, labels, lengths, features = map(lambda x: x.to(self.device), batch)
                if self.args.feature_used == "all":
                    pred = self.dan(notes, lengths, feature=features)
                else:
                    pred = self.dan(notes, lengths)
                prob = torch.softmax(pred, dim=1)[:,1]
                probs.extend([p.item() for p in prob])
        return probs
