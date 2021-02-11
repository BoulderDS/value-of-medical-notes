#!/bin/bash

#python -m in_hospital_mortality.linear_model.main_combined --period_length 24 --data /data/physician_notes/new_experiments/in_hospital_mortality_24/ > out_24.txt
#python -m in_hospital_mortality.linear_model.main_combined --period_length 48 --data /data/physician_notes/new_experiments/in_hospital_mortality_48/ > out_48.txt
#python -m in_hospital_mortality.linear_model.main_combined --period_length -1 --data /data/physician_notes/new_experiments/in_hospital_mortality_retro/ > out_retro.txt

#python -m in_hospital_mortality.linear_model.subset_combined --period_length 24 --data /data/physician_notes/new_experiments/in_hospital_mortality_24/ > out_subset_24.txt
#python -m in_hospital_mortality.linear_model.subset_combined --period_length 48 --data /data/physician_notes/new_experiments/in_hospital_mortality_48/ > out_subset_48.txt
#python -m in_hospital_mortality.linear_model.subset_combined --period_length -1 --data /data/physician_notes/new_experiments/in_hospital_mortality_retro/ > out_subset_retro.txt

#python -m in_hospital_mortality.linear_model.subset_similarity --period_length 24 --data /data/physician_notes/new_experiments/in_hospital_mortality_24/ > out_subset_simi_24.txt
#python -m in_hospital_mortality.linear_model.subset_similarity --period_length 48 --data /data/physician_notes/new_experiments/in_hospital_mortality_48/ > out_subset_simi_48.txt
#python -m in_hospital_mortality.linear_model.subset_similarity --period_length -1 --data /data/physician_notes/new_experiments/in_hospital_mortality_retro/ > out_subset_simi_retro.txt

python -m in_hospital_mortality.linear_model.subset_language --period_length 24 --data /data/physician_notes/new_experiments/in_hospital_mortality_24/ > out_subset_lang_24.txt
python -m in_hospital_mortality.linear_model.subset_language --period_length 48 --data /data/physician_notes/new_experiments/in_hospital_mortality_48/ > out_subset_lang_48.txt
python -m in_hospital_mortality.linear_model.subset_language --period_length -1 --data /data/physician_notes/new_experiments/in_hospital_mortality_retro/ > out_subset_lang_retro.txt
