import multiprocessing
import os
import numpy as np
import pandas as pd
import argparse
import json

from mimic3preprocess.feature_extractor import *

def convert_to_dict(data, header, channel_info):
    '''
    convert data from readers output in to array of arrays format
    from [num_events, num_itemID + 1 (hour)] to [num_itemID, tuple(hour, value)]
    '''
    ret = [[] for i in range(data.shape[1] - 1)]
    # The first column is hours
    for i in range(1, data.shape[1]):
        ret[i-1] = [(t, x) for (t, x) in zip(data[:, 0], data[:, i]) if x != '']
        channel = header[i]
        try:
            if len(channel_info[channel]['possible_values']) != 0:
                ret[i-1] = list(map(lambda x: (x[0], channel_info[channel]['values'][x[1]]), ret[i-1]))
        except KeyError:
            ret[i-1] = list(map(lambda x: (x[0], 0), ret[i-1]))
        try:
            ret[i-1] = list(map(lambda x: (float(x[0]), float(x[1])), ret[i-1]))
        except ValueError:
            ret[i-1] = list(map(lambda x: (float(x[0]), float(0)), ret[i-1]))
    return ret
def read_timeseries(path, ts_filename):

    note_ranges = ['900001','900002','900003','900004','900005','900006','900007','900008','900010','900011','900012','900013','900014','900015', '900016']
    try:
        file_name = os.path.join(path, "train", ts_filename)
        tsfile_df = pd.read_csv(file_name, low_memory = False)
    except:
        file_name = os.path.join(path, "test", ts_filename)
        tsfile_df = pd.read_csv(file_name, low_memory = False)

    tsfile_df = tsfile_df.replace(np.nan, '', regex = True)
    hours = tsfile_df.pop('Hours')
    CGID = tsfile_df.pop('CGID')
    tsfile_structured = tsfile_df.drop(columns = note_ranges)
    tsfile_note = tsfile_df[note_ranges]
    tsfile_structured = tsfile_structured.reindex(sorted(tsfile_structured.columns, key=lambda x: int(x)), axis=1)
    tsfile_note = tsfile_note.reindex(sorted(tsfile_note.columns, key=lambda x: int(x)), axis = 1)
    tsfile_structured.insert(0, 'Hours', hours)
    tsfile_note.insert(0,'Hours', hours)
    return (tsfile_structured.values, tsfile_note.values, tsfile_structured.columns, tsfile_note.columns)

def extract_features_from_rawdata(chunk, header, period, channel_info):
    """
    para:
        chunk: an episode [num_events, num_itemID + 1 (hour)]
    """
    data = convert_to_dict(chunk, header, channel_info)
    return extract_features_single_episode(data, period)

def subprocess(listfile, args):
    with open(os.path.join(os.path.dirname(__file__), "../../in_hospital_mortality/resources/channel_info.json")) as channel_info_file:
        channel_info = json.loads(channel_info_file.read())
    for f in listfile:
        (structured_data, note, structured_header, note_header) = read_timeseries(args.data, f)
        feature = extract_features_from_rawdata(structured_data, structured_header, args.period_length, channel_info)
        np.save(os.path.join(args.output_dir, f[:-4]+".npy"), feature)
        print(feature.shape)
        print("save file", f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--period_length', type=float, default=24, help='specify the period of prediction, -1 denotes retrospective',
                        choices=[24, 48, -1])
    parser.add_argument('--num_worker', type=int, default=40, help='number of cores to use')
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                        default='/data/joe/physician_notes/mimic-data/in_hospital_retro/')
    parser.add_argument('--output_dir', type=str, help = 'Path to the directory to store all structured features of admissions',
                        default='/data/joe/physician_notes/mimic-data/preprocessed/features_24/')
    args = parser.parse_args()
    print(args)
    train_list = pd.read_csv(os.path.join(args.data, "train", "listfile.csv"))
    test_list = pd.read_csv(os.path.join(args.data, "test", "listfile.csv"))
    all_files = list(train_list['stay']) + list(test_list['stay'])
    print("Number of files:", len(all_files))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    num_files = len(all_files)
    chunk_size = num_files//args.num_worker
    for i in range(args.num_worker+1):
        ## right here
        p = multiprocessing.Process(target=subprocess, args=(all_files[chunk_size*i:chunk_size*(i+1)], args))
        p.start()
        print(f"enter worker: {i}")
