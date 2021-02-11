import multiprocessing
import os
import numpy as np
import pandas as pd
import argparse
import json

def convert_to_dict(data, header, channel_info):
    '''
    convert data from readers output in to array of arrays format
    @params
        data: [num_events, num_itemID]
    '''
    for col in range(0, data.shape[1]):
        channel = header[col]
        for row in range(0, data.shape[0]):
            if pd.isnull(data[row][col]):
                continue
            try:
                if len(channel_info[channel]['possible_values']) != 0:
                    data[row][col] = channel_info[channel]['values'][data[row][col]]
            except KeyError:
                    data[row][col] = 0
            try:
                data[row][col] = float(data[row][col])
            except ValueError:
                data[row][col] = float(0)
    return data

def read_timeseries(path, ts_filename, n_hours):
    if n_hours == -1:
        n_hours = float('inf')

    note_ranges = ['900001','900002','900003','900004','900005','900006','900007','900008','900010','900011','900012','900013','900014','900015', '900016']
    try:
        file_name = os.path.join(path, "train", ts_filename[0], ts_filename[1])
        tsfile_df = pd.read_csv(file_name, low_memory = False)
    except:
        file_name = os.path.join(path, "test", ts_filename[0], ts_filename[1])
        tsfile_df = pd.read_csv(file_name, low_memory = False)

    #tsfile_df = tsfile_df.replace(np.nan, '', regex = True)
    # select period
    # add hours to discharge note event
    if pd.isnull(tsfile_df['Hours'].iloc[-1]):
        if len(tsfile_df['Hours']) >1:
            tsfile_df['Hours'].iloc[-1] = tsfile_df['Hours'].iloc[-2]

    tsfile_df = tsfile_df[(tsfile_df['Hours'] > 1e-6) & (tsfile_df['Hours'] < n_hours + 1e-6)].reset_index(drop=True)
    hours = tsfile_df.pop('Hours')
    CGID = tsfile_df.pop('CGID')
    tsfile_note = tsfile_df[note_ranges] # pop all notes
    tsfile_note = tsfile_note.reindex(sorted(tsfile_note.columns, key=lambda x: int(x)), axis=1)
    tsfile_structured = tsfile_df.drop(columns = note_ranges)
    tsfile_structured = tsfile_structured.reindex(sorted(tsfile_structured.columns, key=lambda x: int(x)), axis=1)

    # building mask. 1 => having value; 0=> missing value
    structured_mask = 1 - pd.isnull(tsfile_structured).values.astype(int)
    note_mask =  1 - pd.isnull(tsfile_note).values.astype(int)
    tsfile_note.insert(0,'Hours', hours)

    return (hours.values, tsfile_structured.values, structured_mask, tsfile_structured.columns, tsfile_note, note_mask)

def get_delta(hours, mask):
    """
    Implementation of formula (2) in GRU-D paper
    @params
        hours: [num_events]
        mask: [num_events, num_itemID]
    @return
        delta: [num_events, num_itemID]
    """
    delta = np.zeros(mask.shape, dtype=float)
    # first delta is 0, start at 1
    for i in range(1, mask.shape[0]):
        pre_delta = delta[i-1,:]
        pre_delta = pre_delta * (1-mask[i,:]) # previous delta would be zero if mask == 1 (having value)
        delta_hour = hours[i]-hours[i-1]
        delta[i] = delta_hour + pre_delta # s_t - s_{t-1} + (delta), delta_hour will broadcast to all items
    return delta

def get_last_observed(data, mask):
    """
    Implementation of getting last observed x
    @params
        data: [num_events, num_itemID]
        mask: [num_events, num_itemID]
    @return
        last_observed_x: [num_events, num_itemID]
    """
    data = np.nan_to_num(data.astype(float))
    last_observed = np.zeros(mask.shape, dtype=float)
    last_observed[0] = data[0]
    for i in range(1, mask.shape[0]):
        pre_observed = last_observed[i-1,:]
        cur_observed = data[i,:]
        pre_observed = pre_observed * (1-mask[i,:]) # use previous observed
        cur_observed = cur_observed * mask[i,:] # use current observed
        last_observed[i] = pre_observed+cur_observed
    return last_observed

def subprocess(listfile, args):
    with open(os.path.join(os.path.dirname(__file__), "../../in_hospital_mortality/resources/channel_info.json")) as channel_info_file:
        channel_info = json.loads(channel_info_file.read())
    for f in listfile:
        (hours, structured_data, structured_mask, header, note, note_mask) = read_timeseries(args.data, f, args.period_length)
        if hours.shape[0] == 0:
            continue
        #print("shape", hours.shape, structured_data.shape, structured_mask.shape, note_mask.shape)
        name = f[0]+"_"+f[1]
        structured_data = convert_to_dict(structured_data, header, channel_info)
        structured_delta = get_delta(hours, structured_mask)
        structured_last_observed = get_last_observed(structured_data, structured_mask)
        note_delta = get_delta(hours, note_mask)
        np.save(os.path.join(args.output_dir, 'structured_data', name[:-4]+".npy"), structured_data)
        np.save(os.path.join(args.output_dir, 'structured_mask', name[:-4]+".npy"), structured_mask)
        np.save(os.path.join(args.output_dir, 'structured_delta', name[:-4]+".npy"), structured_delta)
        np.save(os.path.join(args.output_dir, 'structured_last_observed', name[:-4]+".npy"), structured_last_observed)
        note.to_csv(os.path.join(args.output_dir, 'note', name[:-4]+".csv"))
        np.save(os.path.join(args.output_dir, 'note_mask', name[:-4]+".npy"), note_mask)
        np.save(os.path.join(args.output_dir, 'note_delta', name[:-4]+".npy"), note_delta)
        print("save file", f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--period_length', type=float, default=24, help='specify the period of prediction, -1 denotes retrospective',
                        choices=[24, 48, -1])
    parser.add_argument('--num_worker', type=int, default=1, help='number of cores to use')
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                        default='/data/joe/physician_notes/mimic-data/preprocessed/')
    parser.add_argument('--output_dir', type=str, help = 'Path to the directory to store all structured features of admissions',
                        default='/data/joe/physician_notes/mimic-data/preprocessed/grud_features_24/')
    args = parser.parse_args()
    print(args)
    #train_list = pd.read_csv(os.path.join(args.data, "train", "listfile.csv"))
    #test_list = pd.read_csv(os.path.join(args.data, "test", "listfile.csv"))
    #all_files = list(train_list['stay']) + list(test_list['stay']):wq
    all_files = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.data, 'train'))))
    for (patient_index, patient) in enumerate(patients):
        patient_folder = os.path.join(args.data, 'train', patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))
        for ts_filename in patient_ts_files:
            all_files.append((patient, ts_filename))
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.data, 'test'))))
    for (patient_index, patient) in enumerate(patients):
        patient_folder = os.path.join(args.data, 'test', patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))
        for ts_filename in patient_ts_files:
            all_files.append((patient, ts_filename))
    print("Number of files:", len(all_files))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.makedirs(os.path.join(args.output_dir, 'structured_data'))
        os.makedirs(os.path.join(args.output_dir, 'structured_mask'))
        os.makedirs(os.path.join(args.output_dir, 'structured_delta'))
        os.makedirs(os.path.join(args.output_dir, 'structured_last_observed'))
        os.makedirs(os.path.join(args.output_dir, 'note'))
        os.makedirs(os.path.join(args.output_dir, 'note_mask'))
        os.makedirs(os.path.join(args.output_dir, 'note_delta'))
    num_files = len(all_files)
    chunk_size = num_files//args.num_worker
    for i in range(args.num_worker+1):
        ## right here
        p = multiprocessing.Process(target=subprocess, args=(all_files[chunk_size*i:chunk_size*(i+1)], args))
        p.start()
        print(f"enter worker: {i}")

