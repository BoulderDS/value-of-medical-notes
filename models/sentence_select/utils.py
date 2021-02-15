import pickle
from nltk import sent_tokenize
import numpy as np
import spacy
from sklearn.metrics import roc_auc_score, average_precision_score
import os,sys
import pandas as pd
from nltk import word_tokenize
import math
from scipy.spatial import distance
from models import config
from sklearn.metrics.pairwise import cosine_similarity
import re
# from multiprocessing import Pool
# pool = Pool()
nlp = spacy.load('en_core_web_sm', disable=['tokenizer', 'ner', 'textcat'])
medical_terms = set(open('./in_hospital_mortality/resources/wordlist.txt').read().lower().splitlines())
sim_func = {"max":np.amax, "avg":np.mean}
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

def note_filter(notes, from_notes, note_types):
    note_ids = config.note_type[note_types]
    zipped_note = [(n,n_id) for n, n_id in zip(notes, from_notes) if n_id in note_ids]

    if len(zipped_note) == 0:
        return [], []
    notes, from_notes = zip(*zipped_note)
    return [notes[-1]], [from_notes[-1]]

def segmentSentence(model, segment, note, from_notes, datapath, stay, compare=False):
    #note = note.replace("\n", " ")
    
    seg = segment.split("_")[0]
    is_token = "token" in segment
    is_percent = "percent" in segment
    # check if we want to use only discharge
    if not compare and model.args.filter and seg[:10] != "similarity" and seg[:14] != "sentsimilarity":
        note, from_notes = note_filter(note, from_notes, model.args.filter)
        if not note:
            return [], []
    if seg == "all":
        return segmentAll(note, from_notes)
    elif seg == "longest":
        num_sent = int(segment.split("_")[-1])
        return segmentLongestSentence(note, num_sent, from_notes, is_token=is_token, is_percent=is_percent)
    elif seg == "sample":
        num_sent = int(segment.split("_")[-1])
        return segmentSample(note, num_sent, from_notes, is_token=is_token, is_percent=is_percent)
    elif seg == "longestwindow":
        num_sent = int(segment.split("_")[-1])
        return segmentLongestSentenceWindow(note, num_sent, from_notes, is_token=is_token, is_percent=is_percent)
    elif seg == "window":
        window_size = int(segment.split("_")[-1])
        return segmentWindow(note, window_size, from_notes, is_token=is_token, is_percent=is_percent)
    elif seg == "countmedical":
        window_size = int(segment.split("_")[-1])
        return segmentCountMedicalWindow(note, window_size, from_notes, is_token=is_token, is_percent=is_percent)
    elif seg == "fracmedical":
        window_size = int(segment.split("_")[-1])
        return segmentFracMedicalWindow(note, window_size, from_notes, is_token=is_token, is_percent=is_percent)
    elif seg == "countmedicalsent":
        window_size = int(segment.split("_")[-1])
        return segmentCountMedicalSentences(note, window_size, from_notes, is_token=is_token, is_percent=is_percent)
    elif seg == "fracmedicalsent":
        window_size = int(segment.split("_")[-1])
        is_norm = "norm" in segment
        return segmentFracMedicalSentences(note, window_size, from_notes, is_norm=is_norm, is_token=is_token, is_percent=is_percent)
    elif seg == "section":
        return segmentSection(note, datapath, from_notes, is_token=is_token, is_percent=is_percent)
    elif seg == "similarity":
        method = segment.split("_")[1]
        sim_cal = segment.split("_")[2]
        num = int(segment.split("_")[-1])
        is_norm = "norm" in segment
        return segmentSimilarity(model, note, method, sim_cal, num, from_notes, stay, filter=model.args.filter, is_norm=is_norm, is_token=is_token, is_percent=is_percent)
    elif seg == "similaritymix":
        num = int(segment.split("_")[-1])
        sim_ratio = float(segment.split("_")[1])
        sim_cal = segment.split("_")[2]
        is_norm = "norm" in segment
        return segmentSimilarityMix(model, note, sim_ratio, sim_cal, num, from_notes, stay, filter=model.args.filter, is_norm=is_norm, is_token=is_token, is_percent=is_percent)
    elif seg == "sentsimilarity":
        method = segment.split("_")[1]
        sim_cal = segment.split("_")[2]
        num = int(segment.split("_")[-1])
        is_norm = "norm" in segment
        return segmentSentSimilarity(model, note, method, sim_cal, num, from_notes, stay, filter=model.args.filter, is_norm=is_norm, is_token=is_token, is_percent=is_percent)
    elif seg == "sentsimilaritymix":
        num = int(segment.split("_")[-1])
        sim_ratio = float(segment.split("_")[1])
        sim_cal = segment.split("_")[2]
        is_norm = "norm" in segment
        return segmentSentSimilarityMix(model, note, sim_ratio, sim_cal, num, from_notes, stay, filter=model.args.filter, is_norm=is_norm, is_token=is_token, is_percent=is_percent)
    elif segment[:3] == "pos":
        return segmentPOSSentence(note, datapath, from_notes, is_token=is_token, is_percent=is_percent)
    else:
        print(segment)
        print("wrong segment")

def segmentSimilarity(model, note, method, sim_cal, num, from_notes, stay, filter=False, is_norm=False, is_token=False, is_percent=False):
    source_sentences = []
    source_note_ids = []
    source_sent_lengths = []
    target_sentences = []
    target_note_ids = []
    target_sent_lengths = []

    last_note_ids = config.note_type[filter]
    if sum(last_id in from_notes for last_id in last_note_ids) == 0:
        return [],[]

    last_note_idx = -1
    for last_id in last_note_ids:
        if last_id in from_notes:
            tmp_note_idx = len(from_notes) - from_notes[::-1].index(last_id) - 1
            last_note_idx = max(last_note_idx, tmp_note_idx)
    assert last_note_idx != -1

    count = 0
    for i, (n, n_id) in enumerate(zip(note, from_notes)):

        if filter and i == last_note_idx: # last note
            count += 1
            ss = sent_tokenize(n)
            for s in ss:
                sent_len = len(word_tokenize(s))
                if sent_len >= 5:
                    target_sentences.append(s)
                    target_sent_lengths.append(sent_len)
        elif filter:
            source_sentences.append(n)
        else:
            ss = sent_tokenize(n)
            for s in ss:
                sent_len = len(word_tokenize(s))
                if sent_len >= 5:
                    target_sentences.append(s)
                    target_sent_lengths.append(sent_len)
    assert count == 1
    #print(len(sentences), len(discharge_sentences))1
    if filter and (not target_sentences or not source_sentences):
        return [], []
    elif not filter:
        source_sentences = target_sentences

    if not os.path.exists(f'{model.args.data}/select_sentence/similarity/doc_{stay}_{filter}.npy'):
        if model.args.model == "LR":
            tfidf_transformer = dict(model.model.named_steps['union']\
                            .transformer_list).get('tfidf_pipe').named_steps['tfidf']
            source_embeddings = tfidf_transformer.transform({"text":source_sentences}).toarray()
            target_embeddings = tfidf_transformer.transform({"text":target_sentences}).toarray()
        elif model.args.model == "DAN":
            # sentence_embeddings = model.encode(sentences)
            # discharge_embeddings = model.encode(discharge_sentences)
            tfidf_transformer = dict(model.model.named_steps['union']\
                            .transformer_list).get('tfidf_pipe').named_steps['tfidf']
            source_embeddings = tfidf_transformer.transform({"text":source_sentences}).toarray()
            target_embeddings = tfidf_transformer.transform({"text":target_sentences}).toarray()
        #print(sentence_embeddings.shape, discharge_embeddings.shape)
        # [len of discharge * len of previous sents]
        similarity = cosine_similarity(target_embeddings, source_embeddings)
        np.save(f'{model.args.data}/select_sentence/similarity/doc_{stay}_{filter}.npy', similarity)
    else:
        similarity = np.load(f'{model.args.data}/select_sentence/similarity/doc_{stay}_{filter}.npy' , allow_pickle=True)
    #print(distances.shape)
    #distances = np.concatenate(distances, axis=0)
    
    # mean or max similarity with other sents
    if is_norm:
        similarity = sim_func[sim_cal](similarity, axis=1)*np.sqrt(target_sent_lengths) # length normalization
    else:
        similarity = sim_func[sim_cal](similarity, axis=1)

    #print(mean_distances.shape)

    if is_token:
        if method == "near":
            idxs = np.argsort(similarity)[::-1]
        if method == "far":
            idxs = np.argsort(similarity)
        bestsent = get_tokens(target_sentences, num, target_sent_lengths, idxs, is_percent=is_percent)
    else:
        if method == "near":
            idxs = np.argsort(similarity)[::-1][:num]
        if method == "far":
            idxs = np.argsort(similarity)[:num]
        bestsent = " ".join([target_sentences[i] for i in idxs])

    return [bestsent], ["900016" if filter=="discharge" else "900002"]

def segmentSimilarityMix(model, note, sim_ratio, sim_cal, num, from_notes, stay, filter=False, is_norm=False, is_token=False, is_percent=False):
    source_sentences = []
    source_note_ids = []
    source_sent_lengths = []
    target_sentences = []
    target_note_ids = []
    target_sent_lengths = []

    last_note_id = config.note_type[filter][0]
    if last_note_id not in from_notes:
        return [],[]
    last_note_idx = len(from_notes) - from_notes[::-1].index(last_note_id) - 1

    count = 0
    for i, (n, n_id) in enumerate(zip(note, from_notes)):

        if filter and i == last_note_idx: # last note
            count += 1
            ss = sent_tokenize(n)
            for s in ss:
                sent_len = len(word_tokenize(s))
                if sent_len >= 5:
                    target_sentences.append(s)
                    target_sent_lengths.append(sent_len)
        elif filter:
            source_sentences.append(n)
        else:
            ss = sent_tokenize(n)
            for s in ss:
                sent_len = len(word_tokenize(s))
                if sent_len >= 5:
                    target_sentences.append(s)
                    target_sent_lengths.append(sent_len)
    assert count == 1
    #print(len(sentences), len(discharge_sentences))1
    if filter and (not target_sentences or not source_sentences):
        return [], []
    elif not filter:
        source_sentences = target_sentences
    if not os.path.exists(f'{model.args.data}/select_sentence/similarity/doc_{stay}_{filter}.npy'):
        if model.args.model == "LR":
            tfidf_transformer = dict(model.model.named_steps['union']\
                            .transformer_list).get('tfidf_pipe').named_steps['tfidf']
            source_embeddings = tfidf_transformer.transform({"text":source_sentences}).toarray()
            target_embeddings = tfidf_transformer.transform({"text":target_sentences}).toarray()
        elif model.args.model == "DAN":
            # sentence_embeddings = model.encode(sentences)
            # discharge_embeddings = model.encode(discharge_sentences)
            tfidf_transformer = dict(model.model.named_steps['union']\
                            .transformer_list).get('tfidf_pipe').named_steps['tfidf']
            source_embeddings = tfidf_transformer.transform({"text":source_sentences}).toarray()
            target_embeddings = tfidf_transformer.transform({"text":target_sentences}).toarray()
        #print(sentence_embeddings.shape, discharge_embeddings.shape)
        # [len of discharge * len of previous sents]
        similarity = cosine_similarity(target_embeddings, source_embeddings)
        np.save(f'{model.args.data}/select_sentence/similarity/doc_{stay}_{filter}.npy', similarity)
    else:
        similarity = np.load(f'{model.args.data}/select_sentence/similarity/doc_{stay}_{filter}.npy' , allow_pickle=True)
    #print(distances.shape)
    #distances = np.concatenate(distances, axis=0)
    
    # mean or max similarity with other sents
    if is_norm:
        similarity = sim_func[sim_cal](similarity, axis=1)*np.sqrt(target_sent_lengths) # length normalization
    else:
        similarity = sim_func[sim_cal](similarity, axis=1)
    #print(mean_distances.shape)
    if is_token:
        total_len = sum(target_sent_lengths)

        output_sents = []
        #similar
        sim_num = sim_ratio*num
        idxs = np.argsort(similarity)[::-1]
        output_sents.append(get_tokens(target_sentences,sim_num,target_sent_lengths,idxs, is_percent=is_percent))


        #dissimilar
        dis_num = (1-sim_ratio)*num
        if not is_percent:
            dis_num = min(dis_num, total_len-sim_num)
        idxs = np.argsort(similarity)
        output_sents.append(get_tokens(target_sentences,dis_num,target_sent_lengths,idxs, is_percent=is_percent))

    else:
        output_sents = []
        #similar
        sim_num = round(sim_ratio*num)
        idxs = np.argsort(similarity)[::-1][:sim_num]
        output_sents.extend([target_sentences[i] for i in idxs])

        #dissimilar
        dis_num = round((1-sim_ratio)*num)
        idxs = np.argsort(similarity)[:dis_num]
        output_sents.extend([target_sentences[i] for i in idxs])
    #assert len(output_sents) == num
    return [" ".join(output_sents)], [["900016" if filter=="discharge" else "900002"]]

def segmentSentSimilarity(model, note, method, sim_cal, num, from_notes, stay, filter=False, is_norm=False, is_token=False, is_percent=False):
    source_sentences = []
    source_note_ids = []
    source_sent_lengths = []
    target_sentences = []
    target_note_ids = []
    target_sent_lengths = []

    last_note_ids = config.note_type[filter]
    if sum(last_id in from_notes for last_id in last_note_ids) == 0:
        return [],[]

    last_note_idx = -1
    for last_id in last_note_ids:
        if last_id in from_notes:
            tmp_note_idx = len(from_notes) - from_notes[::-1].index(last_id) - 1
            last_note_idx = max(last_note_idx, tmp_note_idx)
    assert last_note_idx != -1

    count = 0
    for i, (n, n_id) in enumerate(zip(note, from_notes)):

        if filter and i == last_note_idx: # last note
            count += 1
            ss = sent_tokenize(n)
            for s in ss:
                sent_len = len(word_tokenize(s))
                if sent_len >= 5:
                    target_sentences.append(s)
                    target_sent_lengths.append(sent_len)
        elif filter:
            ss = sent_tokenize(n)
            for s in ss:
                sent_len = len(word_tokenize(s))
                if sent_len >= 10:
                    source_sentences.append(s)
        else:
            ss = sent_tokenize(n)
            for s in ss:
                sent_len = len(word_tokenize(s))
                if sent_len >= 5:
                    target_sentences.append(s)
                    target_sent_lengths.append(sent_len)
    assert count == 1
    #print(len(sentences), len(discharge_sentences))1
    if filter and (not target_sentences or not source_sentences):
        return [], []
    elif not filter:
        source_sentences = target_sentences
    if not os.path.exists(f'{model.args.data}/select_sentence/similarity/sent_{stay}_{filter}.npy'):
        if model.args.model == "LR":
            tfidf_transformer = dict(model.model.named_steps['union']\
                            .transformer_list).get('tfidf_pipe').named_steps['tfidf']
            source_embeddings = tfidf_transformer.transform({"text":source_sentences}).toarray()
            target_embeddings = tfidf_transformer.transform({"text":target_sentences}).toarray()
        elif model.args.model == "DAN":
            # sentence_embeddings = model.encode(sentences)
            # discharge_embeddings = model.encode(discharge_sentences)
            tfidf_transformer = dict(model.model.named_steps['union']\
                            .transformer_list).get('tfidf_pipe').named_steps['tfidf']
            source_embeddings = tfidf_transformer.transform({"text":source_sentences}).toarray()
            target_embeddings = tfidf_transformer.transform({"text":target_sentences}).toarray()
        #print(sentence_embeddings.shape, discharge_embeddings.shape)
        # [len of discharge * len of previous sents]
        similarity = cosine_similarity(target_embeddings, source_embeddings)
        np.save(f'{model.args.data}/select_sentence/similarity/sent_{stay}_{filter}.npy', similarity)
    else:
        similarity = np.load(f'{model.args.data}/select_sentence/similarity/sent_{stay}_{filter}.npy' , allow_pickle=True)
    #print(distances.shape)
    #distances = np.concatenate(distances, axis=0)
    
    # mean or max similarity with other sents
    if is_norm:
        similarity = sim_func[sim_cal](similarity, axis=1)*np.sqrt(target_sent_lengths) # length normalization
    else:
        similarity = sim_func[sim_cal](similarity, axis=1)

    #print(mean_distances.shape)

    if is_token:
        if method == "near":
            idxs = np.argsort(similarity)[::-1]
        if method == "far":
            idxs = np.argsort(similarity)
        bestsent = get_tokens(target_sentences, num, target_sent_lengths, idxs, is_percent=is_percent)
    else:
        if method == "near":
            idxs = np.argsort(similarity)[::-1][:num]
        if method == "far":
            idxs = np.argsort(similarity)[:num]
        bestsent = " ".join([target_sentences[i] for i in idxs])

    return [bestsent], ["900016" if filter=="discharge" else "900002"]

def segmentSentSimilarityMix(model, note, sim_ratio, sim_cal, num, from_notes, stay, filter=False, is_norm=False, is_token=False, is_percent=False):
    source_sentences = []
    source_note_ids = []
    source_sent_lengths = []
    target_sentences = []
    target_note_ids = []
    target_sent_lengths = []

    last_note_id = config.note_type[filter][0]
    if last_note_id not in from_notes:
        return [],[]
    last_note_idx = len(from_notes) - from_notes[::-1].index(last_note_id) - 1

    count = 0
    for i, (n, n_id) in enumerate(zip(note, from_notes)):
        if filter and i == last_note_idx: # last note
            count += 1
            ss = sent_tokenize(n)
            for s in ss:
                sent_len = len(word_tokenize(s))
                if sent_len >= 5:
                    target_sentences.append(s)
                    target_sent_lengths.append(sent_len)
        elif filter:
            ss = sent_tokenize(n)
            for s in ss:
                sent_len = len(word_tokenize(s))
                if sent_len >= 5:
                    source_sentences.append(s)
        else:
            ss = sent_tokenize(n)
            for s in ss:
                sent_len = len(word_tokenize(s))
                if sent_len >= 5:
                    target_sentences.append(s)
                    target_sent_lengths.append(sent_len)
    assert count == 1

    #print(len(sentences), len(discharge_sentences))1
    if filter and (not target_sentences or not source_sentences):
        return [], []
    elif not filter:
        source_sentences = target_sentences

    if not os.path.exists(f'{model.args.data}/select_sentence/similarity/select_sentence/similarity/sent_{stay}_{filter}.npy'):
        if model.args.model == "LR":
            tfidf_transformer = dict(model.model.named_steps['union']\
                            .transformer_list).get('tfidf_pipe').named_steps['tfidf']
            source_embeddings = tfidf_transformer.transform({"text":source_sentences}).toarray()
            target_embeddings = tfidf_transformer.transform({"text":target_sentences}).toarray()
        elif model.args.model == "DAN":
            # sentence_embeddings = model.encode(sentences)
            # discharge_embeddings = model.encode(discharge_sentences)
            tfidf_transformer = dict(model.model.named_steps['union']\
                            .transformer_list).get('tfidf_pipe').named_steps['tfidf']
            source_embeddings = tfidf_transformer.transform({"text":source_sentences}).toarray()
            target_embeddings = tfidf_transformer.transform({"text":target_sentences}).toarray()
        #print(sentence_embeddings.shape, discharge_embeddings.shape)
        # [len of discharge * len of previous sents]
        similarity = cosine_similarity(target_embeddings, source_embeddings)
        np.save(f'{model.args.data}/select_sentence/similarity/select_sentence/similarity/sent_{stay}_{filter}.npy', similarity)
    else:
        similarity = np.load(f'{model.args.data}/select_sentence/similarity/select_sentence/similarity/sent_{stay}_{filter}.npy' , allow_pickle=True)
    #print(distances.shape)
    #distances = np.concatenate(distances, axis=0)
    
    # mean or max similarity with other sents
    if is_norm:
        similarity = sim_func[sim_cal](similarity, axis=1)*np.sqrt(target_sent_lengths) # length normalization
    else:
        similarity = sim_func[sim_cal](similarity, axis=1)
    #print(mean_distances.shape)
    if is_token:
        total_len = sum(target_sent_lengths)

        output_sents = []
        #similar
        sim_num = sim_ratio*num
        idxs = np.argsort(similarity)[::-1]
        output_sents.append(get_tokens(target_sentences,sim_num,target_sent_lengths,idxs, is_percent=is_percent))


        #dissimilar
        dis_num = (1-sim_ratio)*num
        if not is_percent:
            dis_num = min(dis_num, total_len-sim_num)
        idxs = np.argsort(similarity)
        output_sents.append(get_tokens(target_sentences,dis_num,target_sent_lengths,idxs, is_percent=is_percent))

    else:
        output_sents = []
        #similar
        sim_num = round(sim_ratio*num)
        idxs = np.argsort(similarity)[::-1][:sim_num]
        output_sents.extend([target_sentences[i] for i in idxs])

        #dissimilar
        dis_num = round((1-sim_ratio)*num)
        idxs = np.argsort(similarity)[:dis_num]
        output_sents.extend([target_sentences[i] for i in idxs])
    #assert len(output_sents) == num
    return [" ".join(output_sents)], [["900016" if filter=="discharge" else "900002"]]


def segmentAll(note, from_notes):

    bestsent = " ".join(note)
    best_note_ids = " ".join(set(from_notes))
    
    return [bestsent], [best_note_ids]


def segmentLongestSentence(note, num, from_notes, is_token=False, is_percent=False):
    sentences = []
    note_ids = []
    for n, n_id in zip(note, from_notes):
        tmp = sent_tokenize(n)
        sentences.extend(tmp)
        note_ids.extend([n_id]*len(tmp))

    lengths = [len(word_tokenize(s)) for s in sentences]
    #idx = np.argmax(lengths)
    if is_token:
        idxs = np.argsort(lengths)[::-1]
        bestsent, best_note_ids = get_tokens(sentences,num,lengths,idxs,note_ids, is_percent=is_percent)
    else:
        idxs = np.argsort(lengths)[::-1][:num]
        bestsent = " ".join([sentences[i] for i in idxs])
        best_note_ids = " ".join(set([note_ids[i] for i in idxs]))
    
    return [bestsent], [best_note_ids]

def segmentSample(note, num, from_notes, is_token=False, is_percent=False):
    sentences = []
    note_ids = []
    for n, n_id in zip(note, from_notes):
        tmp = sent_tokenize(n)
        sentences.extend(tmp)
        note_ids.extend([n_id]*len(tmp))

    lengths = [len(word_tokenize(s)) for s in sentences]
    #idx = np.argmax(lengths)
    if is_token:
        if len(sentences) == 0:
            return [""],["0"]
        idxs = np.arange(len(lengths))
        np.random.shuffle(idxs)
        bestsent, best_note_ids = get_tokens(sentences,num,lengths,idxs,note_ids, is_percent=is_percent)
    else:
        idxs = np.arange(len(lengths))
        np.random.shuffle(idxs)
        idxs = idxs[:num]
        bestsent = " ".join([sentences[i] for i in idxs])
        best_note_ids = " ".join(set([note_ids[i] for i in idxs]))
    
    return [bestsent], [best_note_ids]

def segmentLongestSentenceWindow(note, num_sent, from_notes, is_token=False, is_percent=False):
    sentences = []
    note_ids = []
    for n, n_id in zip(note, from_notes):
        tmp = sent_tokenize(n)
        sentences.extend(tmp)
        note_ids.extend([n_id]*len(tmp))

    lengths = [len(s) for s in sentences]
    #idx = np.argmax(lengths)
    idxs = np.argsort(lengths)[::-1][0]
    bestsent = " ".join(sentences[int(idxs)-((num_sent+1)//2)+1: int(idxs) + (num_sent//2)])
    best_note_ids = " ".join(set(note_ids[int(idxs)-((num_sent+1)//2)+1: int(idxs) + (num_sent//2)]))
    return [bestsent], [best_note_ids]

def segmentCountMedicalSentences(note, num, from_notes, is_token=False, is_percent=False):
    note_ids = []
    window_sentences = []
    num_medical_term = []
    lengths = []
    for n, n_id in zip(note, from_notes):
        sentences = sent_tokenize(n)
    
        for s in sentences:
            tokens = word_tokenize(s)
            tmp_n = sum([1 for w in tokens if w.lower() in medical_terms])
            lengths.append(len(tokens))
            num_medical_term.append(tmp_n)
            window_sentences.append(s)
            note_ids.append(n_id)

    assert len(window_sentences) == len(num_medical_term)
    assert len(window_sentences) == len(note_ids)

    if len(num_medical_term) == 0:
        return [], []
    if is_token:
        idxs = np.argsort(num_medical_term)[::-1]
        bestsent, best_note_ids = get_tokens(window_sentences,num,lengths,idxs,note_ids, is_percent=is_percent)

    else:
        idxs = np.argsort(num_medical_term)[::-1][:num]
            
        bestsent = " ".join([window_sentences[i] for i in idxs])
        best_note_ids = " ".join(set([note_ids[i] for i in idxs]))
    return  [bestsent], [best_note_ids]

def segmentFracMedicalSentences(note, num, from_notes, is_norm=False, is_token=False, is_percent=False):
    note_ids = []
    window_sentences = []
    num_medical_term = []
    lengths = []
    for n, n_id in zip(note, from_notes):
        sentences = sent_tokenize(n)
    
        for s in sentences:
            tokens = word_tokenize(s)
            tmp_n = sum([1 for w in tokens if w.lower() in medical_terms])
            lengths.append(len(tokens))
            if is_norm:
                num_medical_term.append(tmp_n/len(tokens) * math.sqrt(len(tokens)))
            else:
                num_medical_term.append(tmp_n/len(tokens))
            window_sentences.append(s)
            note_ids.append(n_id)

    assert len(window_sentences) == len(num_medical_term)
    assert len(window_sentences) == len(note_ids)

    if len(num_medical_term) == 0:
        return [], []
    if is_token:
        idxs = np.argsort(num_medical_term)[::-1]
        bestsent, best_note_ids = get_tokens(window_sentences,num,lengths,idxs,note_ids, is_percent=is_percent)
    else:
        idxs = np.argsort(num_medical_term)[::-1][:num]
            
        bestsent = " ".join([window_sentences[i] for i in idxs])
        best_note_ids = " ".join(set([note_ids[i] for i in idxs]))
    return  [bestsent], [best_note_ids]

    
def segmentCountMedicalWindow(note, w_size, from_notes, is_token=False, is_percent=False):
    note_ids = []
    window_sentences = []
    num_medical_term_w = []
    window_lengths = []
    for n, n_id in zip(note, from_notes):
        num_medical_term = []
        lengths = []
        sentences = sent_tokenize(n)
    
        for s in sentences:
            tokens = word_tokenize(s)
            tmp_n = sum([1 for w in tokens if w.lower() in medical_terms])
            lengths.append(len(tokens))
            num_medical_term.append(tmp_n)

        window = [" ".join(sentences[i:i+w_size]) for i in range(len(sentences)-w_size+1)]
        window_sentences.extend(window)
        num_medical_term_w.extend([sum(num_medical_term[i:i+w_size]) for i in range(len(sentences)-w_size+1)])
        window_lengths.extend([sum(lengths[i:i+w_size]) for i in range(len(sentences)-w_size+1)])
        note_ids.extend([n_id]*len(window))

    assert len(window_sentences) == len(num_medical_term_w)
    assert len(window_sentences) == len(note_ids)

    if len(num_medical_term_w) == 0:
        for n, n_id in zip(note, from_notes):
            num_medical_term_w = []
            lens = []
            sentences = sent_tokenize(n)
        
            for s in sentences:
                tokens = word_tokenize(s)
                tmp_n = sum([1 for w in tokens if w.lower() in medical_terms])
                num_medical_term.append(tmp_n)
                lens.append(len(tokens))

            window_sentences.extend([" ".join(sentences) ])
            num_medical_term_w.extend([sum(num_medical_term)])
            note_ids.extend([n_id])

        if len(num_medical_term_w) >0:
            idxs = np.argsort(num_medical_term_w)[::-1][0]
            return [window_sentences[idxs]], [note_ids[idxs]]

        return [""], ["0"]
    idxs = np.argsort(num_medical_term_w)[::-1][0]
    return [window_sentences[idxs]], [note_ids[idxs]]

def segmentFracMedicalWindow(note, w_size, from_notes, is_token=False, is_percent=False):
    note_ids = []
    window_sentences = []
    frac_medical_term_w = []
    for n, n_id in zip(note, from_notes):
        num_medical_term = []
        num_medical_term_w = []
        lens = []
        sentences = sent_tokenize(n)
    
        for s in sentences:
            tokens = word_tokenize(s)
            tmp_n = sum([1 for w in tokens if w.lower() in medical_terms])
            num_medical_term.append(tmp_n)
            lens.append(len(tokens))

        window_sentences.extend([" ".join(sentences[i:i+w_size]) for i in range(len(sentences)-w_size+1)])
        num_medical_term_w = [sum(num_medical_term[i:i+w_size]) for i in range(len(sentences)-w_size+1)]
        lens_w = [sum(lens[i:i+w_size]) for i in range(len(sentences)-w_size+1)]
        frac_medical_term_w.extend([ n/l * math.sqrt(l) for n, l in zip(num_medical_term_w, lens_w)])
        note_ids.extend([n_id]*len(lens_w))

    assert len(window_sentences) == len(frac_medical_term_w)
    assert len(window_sentences) == len(note_ids)

    if len(frac_medical_term_w) == 0:
        for n, n_id in zip(note, from_notes):
            num_medical_term_w = []
            lens = []
            sentences = sent_tokenize(n)
        
            for s in sentences:
                tokens = word_tokenize(s)
                tmp_n = sum([1 for w in tokens if w.lower() in medical_terms])
                num_medical_term.append(tmp_n)
                lens.append(len(tokens))

            window_sentences.extend([" ".join(sentences) ])
            num_medical_term_w = sum(num_medical_term)
            lens_w = sum(lens) if sum(lens) != 0 else 1
            frac_medical_term_w.extend([ num_medical_term_w/lens_w * math.sqrt(lens_w)])
            note_ids.extend([n_id])

        if len(frac_medical_term_w) >0:
            idxs = np.argsort(frac_medical_term_w)[::-1][0]
            return [window_sentences[idxs]], [note_ids[idxs]]
        return [""], ["0"]
    idxs = np.argsort(frac_medical_term_w)[::-1][0]
    return [window_sentences[idxs]], [note_ids[idxs]]

def segmentSection(note, w_size, from_notes, is_token=False, is_percent=False):
    sentences = []
    note_ids = []
    for n, n_id in zip(note, from_notes):
        if n_id == "900002":
            idx = n.find("Assessment and Plan", None)
            if idx is not None:
                sentences.append(n[idx:])
                note_ids.append(n_id)
    sentences = " ".join(sentences)
    if sentences:
        return [sentences], ["900002"]
    else:
        return [], []
def segmentWindow(note, w_size, from_notes, is_token=False, is_percent=False):
    window_sentences = []
    note_ids = []
    for n, n_id in zip(note, from_notes):
        sentences = sent_tokenize(n)
        window_sentences.extend([" ".join(sentences[i:i+w_size]) for i in range(len(sentences)-w_size+1)])
        note_ids.extend([n_id]*len(window_sentences))
    return window_sentences, note_ids

def segmentPOSSentence(segment, note, from_notes, is_token=False, is_percent=False):
    sentences = sent_tokenize(note)
    count_list = []
    for sen in sentences:
        count = 0
        doc = nlp(sen[-1000000:])
        for token in doc:
            # count number of specific POS tagger in a sentence
            if token.tag_ == segment.split("_")[-1]:
                count += 1

        count_list.append(count)
    if 'portion' in segment:
        count_list = [i/len(sentences[i]) for i, c in enumerate(count_list)]
    idx = np.argmax(count_list)
    return [sentences[idx]]

def get_tokens(sentences, num, lengths, idxs, note_ids=None, is_percent=False):
    if is_percent:
        num = round(sum(lengths)*num//100)
    else:
        num = round(num)

    out_sents = []
    count_token = 0
    for idx in idxs:
        cur_len = lengths[idx]
        cur_sent = sentences[idx]
        if count_token+cur_len <= num:
            out_sents.append(cur_sent)
            count_token += cur_len
        elif count_token < num:
            trunc_sent = " ".join(word_tokenize(cur_sent)[:num-count_token])
            out_sents.append(trunc_sent)
            count_token += cur_len
        else:
            break
    if not note_ids:
        return " ".join(out_sents)
    else:
        output_ids=[]
        for i, idx in enumerate(idxs):
            if i < len(out_sents):
                output_ids.append(note_ids[idx])
        return " ".join(out_sents), " ".join(set(output_ids))

def impute_scale_features(list_file, period):
    print("Loading features")
    features = pickle.load(open(f'/data/joe/physician_notes/mimic-data/preprocessed/features_{period}.pkl','rb'))
    def load_data(listfile, features):
        stays, fs = [], []
        for stay in listfile['stay']:
            stays.append(stay)
            fs.append(features[stay])
        return stays, fs

    stays, fs = load_data(list_file, features)

    print("feature length:", len(fs[0]))
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import MinMaxScaler
    print("Impute features")
    if not os.path.exists(f'/data/joe/physician_notes/mimic-data/preprocessed/features_imputer_{period}.pkl'):
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean.fit(fs)
        pickle.dump(imp_mean, open(f'/data/joe/physician_notes/mimic-data/preprocessed/features_imputer_{period}.pkl', 'wb'))
    else:
        imp_mean = pickle.load(open(f'/data/joe/physician_notes/mimic-data/preprocessed/features_imputer_{period}.pkl', 'rb'))
    fs = imp_mean.transform(fs)
    print("feature length:", len(fs[0]))
    print("Scale features")
    if not os.path.exists(f'/data/joe/physician_notes/mimic-data/preprocessed/features_scaler_{period}.pkl'):
        scaler = MinMaxScaler()
        scaler.fit(fs)
        pickle.dump(scaler, open(f'/data/joe/physician_notes/mimic-data/preprocessed/features_scaler_{period}.pkl', 'wb'))
    else:
        scaler = pickle.load(open(f'/data/joe/physician_notes/mimic-data/preprocessed/features_scaler_{period}.pkl', 'rb'))
    fs = scaler.transform(fs)
    #print("feature length:", len(fs[0]))
    return fs, len(fs[0])
    #return {key: value for (key, value) in zip(stays, fs)}, len(fs[0])

def precision_at_k(y_label, y_pred, k):
    # precision @ K percent
    rank = list(zip(y_label, y_pred))
    rank.sort(key=lambda x: x[1], reverse=True)
    num_k = len(y_label)*k//100
    return sum(rank[i][0] == 1 for i in range(num_k))/float(num_k)

def findHighestScore(probs, stays, segments, y_label, from_notes, args):

    dic = {}
    for stay, seg, prob, y, note_id in zip(stays, segments, probs, y_label, from_notes):
        #y = 1- y
        if stay not in dic:
            dic[stay] = (prob, y, seg, note_id)
        elif prob*(2*y-1) > dic[stay][0]: # make y from (1,0) to (1, -1)
            dic[stay] = (prob, y , seg, note_id)

    best_scores = []
    best_sentences = []
    label = []
    stays = []
    note_ids = []
    for k, v in dic.items():
        stays.append(k)
        best_scores.append(v[0])
        label.append(v[1])
        best_sentences.append(v[2])
        note_ids.append(v[3])
    print("ROC_AUC")
    print(roc_auc_score(label, best_scores))
    print("PR_AUC")
    print(average_precision_score(label, best_scores))
    p_at_1 = precision_at_k(label, best_scores,1)
    p_at_5 = precision_at_k(label, best_scores,5)
    p_at_10 = precision_at_k(label, best_scores,10)
    print("P@1,5,10", p_at_1, p_at_5, p_at_10)
    opt = args
    model_name = opt.note +'_'+ opt.feature_period + '.chkpt'
    if opt.feature_used == "all":
        model_name = "feature_text_" + model_name
    elif opt.feature_used == "all_but_notes":
        model_name = "feature_" + model_name
    else:
        model_name = "text_" + model_name

    if opt.reverse_label:
        model_name = "reverse_" + model_name
    if args.filter : #and args.note == "all_but_discharge"
        model_name = args.filter+"_"+model_name
    import pathlib

    path = f'./models/sentence_select/results/{args.model}/{args.task}/{args.segment}'
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(path, model_name[:-6]+".csv"), 'w') as f:
        f.write("TYPE,ROCAUC,PRAUC,P@1,P@5,P@10\n")
        f.write(f"test,{roc_auc_score(label, best_scores):.3f},{average_precision_score(label, best_scores):.3f},{p_at_1:.3f},{p_at_5:.3f},{p_at_10:.3f}")

    df = pd.DataFrame({"stay":stays, "y_label":label, "prob": best_scores, "bestSents":best_sentences, "note_id":note_ids})
    path = f'{args.data}/select_sentence/{args.model}/{args.task}/{args.segment}'
    import pathlib
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(path, model_name[:-6]+".csv")
    print(file_path)
    print("output len:", len(df))
    df.to_csv(file_path)

