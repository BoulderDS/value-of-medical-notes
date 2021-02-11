#!/usr/bin/env python3

import os
import argparse

from mimic3preprocess.mimic3tables import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combine data from selected tables in MIMIC into a per-subject csv file')
    parser.add_argument('mimic3_path', type = str, help = 'Directory containing the MIMIC-III csv files.')
    parser.add_argument('output_path', type = str, help = 'Directory where per-subject data should be written.')
    parser.add_argument('-e', '--event_tables', type = str, nargs = '+', help = 'Tables from which events should be read.', default = ['CHARTEVENTS', 'LABEVENTS', 'INPUTEVENTS_CV', 'INPUTEVENTS_MV', 'OUTPUTEVENTS', 'NOTEEVENTS'])
    parser.add_argument('--verbose', '-v', type = int, help = 'Level of verbosity in output.', default = 1)
    parser.add_argument('--n_workers', type = int, help = 'Level of verbosity in output.', default = 32)
    parser.add_argument('--test', action = 'store_true', help = 'Test Run to Process Only 1K subjects and Only 1 Event Table (First argument in list of arguments for event_tables)')
    args = parser.parse_args()
    print(args)
    try:
        os.makedirs(args.output_path)
    except:
        pass
    patients = read_patients_table(args.mimic3_path)
    admissions = read_admissions_table(args.mimic3_path)
    icustays = read_icustays_times_in_icu_and_db(args.mimic3_path)

    if (args.verbose):
        print ('START:\nRead PATIENTS table -> Total Patients = {}\nRead ADMISSIONS table -> Total Admissions in Hospital = {}\n'.format(patients['SUBJECT_ID'].unique().shape[0], admissions['HADM_ID'].unique().shape[0]))

    admissions = merge_on_subject(admissions, patients)
    admissions = merge_on_subject_admits(admissions, icustays)

    if (args.verbose):
        print ('Merged ADMISSIONS, PATIENTS and ICUSTAYS into ADMISSIONS.\n')

    admissions = add_age_to_admits(admissions)
    admissions = filter_admits_on_age(admissions)

    if (args.verbose):
        print ('Removed Patients with Age < 18:\nRead PATIENTS table -> Total Patients = {}\nRead ADMISSIONS table -> Total Admissions in Hospital = {}\n'.format(admissions['SUBJECT_ID'].unique().shape[0], admissions['HADM_ID'].shape[0]))

    admissions.to_csv(os.path.join(args.output_path, 'all_admissions.csv'), index = False)

    if args.test:
        patient_subset = np.random.choice(patients.shape[0], size = 1000)
        patients = patients.iloc[patient_subset]
        admissions = merge_on_subject(admissions, patients[['SUBJECT_ID']])
        args.event_tables = [args.event_tables[0]]
        print ('Using only', admissions.shape[0], 'admissions and only', args.event_tables[0], 'table')

    subjects = admissions['SUBJECT_ID'].unique()
    break_up_admissions_by_subject(admissions, args.output_path, subjects = subjects, verbose = args.verbose)
    admissions = pd.read_csv(os.path.join(args.output_path, 'all_admissions.csv'))
    subjects = admissions['SUBJECT_ID'].unique()

    for table in args.event_tables:
        read_events_table_and_break_up_by_subject(args.mimic3_path, table, args.output_path, subjects_to_keep = subjects, verbose = args.verbose, n_workers = args.n_workers)
