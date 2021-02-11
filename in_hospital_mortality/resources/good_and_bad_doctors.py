#!/usr/bin/env python3

import os
import argparse
import warnings
import pandas as pd
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Reading the mimic tables')
    parser.add_argument('--mimic3_path',type = str, help = 'Directory containing the MIMIC-III csv files.',
                        default = '/data/physician_notes/mimic-data/')
    parser.add_argument('--save_file', type = str, help = 'Directory in which we store the list of good and bad doctors')
    args = parser.parse_args()
    
    warnings.filterwarnings('ignore')

    # Reading Admissions table and getting the hadm_id and the hospital expire flag for each admission
    
    admissions = pd.read_csv(os.path.join(args.mimic3_path, 'ADMISSIONS.csv.gz'))
    admissions = admissions[['HADM_ID','HOSPITAL_EXPIRE_FLAG']]
    admissions.drop_duplicates(inplace=True)
    admissions.reset_index(drop=True, inplace=True)
    
    # Reading noteevents table ->
    # Will remove all Radiology, Echo and ECG notes as they don't have any associated CGID with them
    # We don't use discharge summary so remove them as well
    
    noteevents = pd.read_csv(os.path.join(args.mimic3_path, 'NOTEEVENTS.csv.gz'))
    noteevents = noteevents[~noteevents.CATEGORY.isin(['Discharge summary','Echo','ECG','Radiology'])]
    noteevents = noteevents[noteevents.ISERROR.isna()]
    noteevents.drop(columns = ['ROW_ID','SUBJECT_ID','CHARTDATE','CHARTTIME','STORETIME','DESCRIPTION','ISERROR'], inplace = True)
    noteevents.drop_duplicates(inplace=True)
    noteevents.reset_index(drop=True, inplace = True)
    
    # Combine notes and admissions table

    notes_all = pd.merge(admissions, noteevents, on = 'HADM_ID')

    # Get counts of 0's and 1's from cgid and store the ones with 1's
    # Size get counts of 0's and 1's seperately
    # unstack().fillna(0).stack() fills the value count with 0 if nothing exists in that category
    # Then remove all records with 0 counts

    cgid_1_counts = notes_all.groupby(['CGID','HOSPITAL_EXPIRE_FLAG']).size().unstack().fillna(0).stack().to_frame().reset_index()
    cgid_1_counts = cgid_1_counts[cgid_1_counts['HOSPITAL_EXPIRE_FLAG'] == 1]
    cgid_1_counts.drop(columns = 'HOSPITAL_EXPIRE_FLAG', inplace=True)
    cgid_1_counts.rename(columns = {0:'COUNT_1'}, inplace=True)

    # Get counts of all notes written by a CGID
    
    cgid_all_counts = notes_all.groupby(['CGID']).size().to_frame().reset_index()
    cgid_all_counts.rename(columns = {0: 'COUNT_ALL'}, inplace = True)
    
    # Merge these two tables

    cgid_complete = pd.merge(cgid_1_counts, cgid_all_counts, on = 'CGID')
    cgid_complete['FRACTION_DEAD'] = cgid_complete['COUNT_1'] / cgid_complete['COUNT_ALL']
    cgid_complete.sort_values(by = 'COUNT_ALL', inplace = True)
    
    '''
    If plotting a distribution is required for FRACTION_DEAD, import seaborn and matplotlib
    and write sns.distplot(cgid_complete['FRACTION_DEAD'])
    
    Take the leftmost 25 % (close to 0) as good doctors and rightmost 25 % (close to 1) as bad doctors

    Get the quantiles by using 
    > cgid_complete.quantile(q = 0.25)
    > cgid_complete.quantile(q = 0.75)
    and look at the value in FRACTION_DEAD. Use the same for getting the list of good and bad doctors
    '''

    good_doctors = cgid_complete[cgid_complete['FRACTION_DEAD'] <= 0.011193]['CGID'].values
    bad_doctors = cgid_complete[cgid_complete['FRACTION_DEAD'] >= 0.220509]['CGID'].values
    neutral_doctors = cgid_complete[(~cgid_complete.CGID.isin(good_doctors)) & (~cgid_complete.CGID.isin(bad_doctors))]['CGID'].values

    # Saving all these 3 arrays

    np.save(os.path.join(args.save_file,'good_doctors.npy'), good_doctors)
    np.save(os.path.join(args.save_file,'bad_doctors.npy'), bad_doctors)
    np.save(os.path.join(args.save_file,'neutral_doctors.npy'), neutral_doctors)
