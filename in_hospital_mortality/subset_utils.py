#!/usr/bin/env python3

import numpy as np
import pandas as pd
import spacy

from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cdist

def create_seperate_note_cat(chunk):

    nursing_other = [] 
    physician = [] 
    nutrition = [] 
    general = [] 
    nursing = [] 
    respiratory = [] 
    rehab_services = [] 
    social_work = [] 
    echo = [] 
    ecg = [] 
    case_management = [] 
    pharmacy = [] 
    consult = [] 
    radiology = [] 
    for note in chunk: 
        nursing_other.append(' '.join(row for row in note[:,1] if row != '')) 
        physician.append(' '.join(row for row in note[:,2] if row != '')) 
        nutrition.append(' '.join(row for row in note[:,3] if row != '')) 
        general.append(' '.join(row for row in note[:,4] if row != '')) 
        nursing.append(' '.join(row for row in note[:,5] if row != '')) 
        respiratory.append(' '.join(row for row in note[:,6] if row != '')) 
        rehab_services.append(' '.join(row for row in note[:,7] if row != '')) 
        social_work.append(' '.join(row for row in note[:,8] if row != '')) 
        echo.append(' '.join(row for row in note[:,9] if row != '')) 
        ecg.append(' '.join(row for row in note[:,10] if row != '')) 
        case_management.append(' '.join(row for row in note[:,11] if row != '')) 
        pharmacy.append(' '.join(row for row in note[:,12] if row != '')) 
        consult.append(' '.join(row for row in note[:,13] if row != '')) 
        radiology.append(' '.join(row for row in note[:,14] if row != ''))

    return nursing_other, physician, nutrition, general, nursing, respiratory, rehab_services, social_work, echo, ecg, case_management, pharmacy, consult, radiology

def create_english_medical_split(chunk):

    with open('in_hospital_mortality/resources/top_10000_english_words.txt') as english_words_file:
        english_words = english_words_file.read().split('\n')
    with open('in_hospital_mortality/resources/medical_words.txt') as medical_words_file:
        medical_words = medical_words_file.read().split('\n')
    note_data = [' '.join(' '.join(x for x in row if x != '') for row in note[:,1:]) for note in chunk]
    english_notes = [' '.join(word for word in note.split() if word.lower() in english_words) for note in note_data]
    medical_notes = [' '.join(word for word in note.split() if word.lower() in medical_words) for note in note_data]
    return english_notes, medical_notes

def create_pos_tag_split(chunk):

    nlp = spacy.load("en_core_web_sm", disable = ['tokenizer','parser','ner','textcat'])
    nlp.max_length = 10000000
    tagger = nlp.create_pipe("tagger")
    note_data = [' '.join(' '.join(x for x in row if x != '') for row in note[:,1:]) for note in chunk]
    noun_notes = []
    proper_noun_notes = []
    adjective_notes = []
    verb_notes = []
    for notes in note_data:
        doc = nlp(notes)
        tagged_note = nlp.tagger(doc)
        note_pos_dict = {'NOUN': '', 'PROPN': '', 'ADJ': '', 'VERB': ''}
        for word in tagged_note:
            try:
                note_pos_dict[word.pos_] += ' ' + word.text
            except KeyError:
                continue
        noun_notes.append(note_pos_dict['NOUN'])
        proper_noun_notes.append(note_pos_dict['PROPN'])
        adjective_notes.append(note_pos_dict['ADJ'])
        verb_notes.append(note_pos_dict['VERB'])

    return noun_notes, proper_noun_notes, adjective_notes, verb_notes

def remove_copy_pasting_notes(chunk):

    nlp = spacy.load('en_core_web_sm', disable = ['tokenizer','ner','tagger','textcat'])
    nlp.max_length = 10000000
    vectorizer = CountVectorizer()
    non_copy_note_data = []
    copy_note_data = []
    similarities = []
    empty_timeseries = np.full(14, '')
    overall_text_mat = []
    overall_similarity_mat = []
    for note in chunk:
        text_mat = []
        sentence_mat = []
        cumulative_note = []
        for row in note:
            text_row = []
            sentence_row = []
            if np.array_equal(empty_timeseries, row[1:]) == False:
                for text in row[1:]:
                    if text == '':
                        continue
                    doc = nlp(text)
                    sentences = [sent.string.strip() for sent in doc.sents]
                    sentences = [sent.strip() for sent in sentences if len(sent.strip()) > 0]
                    cumulative_note.extend(sentences)
                    text_row.append(text)
                    sentence_row.append(sentences)
                text_mat.append(text_row)
                sentence_mat.append(sentence_row)

        if (len(cumulative_note) == 0):
            overall_text_mat.append([])
            overall_similarity_mat.append([])
            continue

        all_notes = vectorizer.fit_transform(cumulative_note)
        similarity_mat = []
        total_length = 0

        for i in range(len(sentence_mat)):
            similarity_row = []
            row_length = 0
            for j in range(len(sentence_mat[i])):
                if i == 0:
                    similarity_row.append(0)
                    similarities.append(0)
                    row_length += len(sentence_mat[i][j])
                    continue
                current_note_x = all_notes[total_length:total_length + len(sentence_mat[i][j])].todense()
                previous_note_x = all_notes[0:total_length].todense()
                similarity = 1 - cdist(current_note_x, previous_note_x, 'cosine')
                similarity[np.isnan(similarity)] = 0
                similarity_row.append(np.mean(np.max(similarity, axis = 1)))
                similarities.append(np.mean(np.max(similarity, axis = 1)))
                row_length += len(sentence_mat[i][j])
            similarity_mat.append(similarity_row)
            total_length += row_length

        overall_text_mat.append(text_mat)
        overall_similarity_mat.append(similarity_mat)                         

    similarities = np.array(similarities)
    similarities = similarities[~np.isnan(similarities)]    
    # 0.25 quantile for notes with low similarities
    # 0.75 quantile for notes with high similarities
    low_similarity_threshold = np.quantile(np.array(similarities), 0.75)
    high_similarity_threshold = np.quantile(np.array(similarities), 0.25)

    for k in range(len(overall_text_mat)):
        copy_note = ''
        non_copy_note = ''
        if overall_text_mat[k]:
            for i in range(len(overall_text_mat[k])):
                for j in range(len(overall_text_mat[k][i])):
                    if overall_similarity_mat[k][i][j] <= low_similarity_threshold:
                        non_copy_note += overall_text_mat[k][i][j] + ' '
                    if overall_similarity_mat[k][i][j] >= high_similarity_threshold:
                        copy_note += overall_text_mat[k][i][j] + ' '
        non_copy_note_data.append(non_copy_note)
        copy_note_data.append(copy_note)

    return non_copy_note_data, copy_note_data

