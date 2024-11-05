f_name="withBP"
activation="tanh"
#dataFolder="/Users/a.e./Dropbox/ANTS_CANTS/transformer_data/ETDataset/ETT-small"
dataFolder="/Users/a.e./Dropbox/ANTS_CANTS/datasets/2019_ngafid_transfer"
INPUT="AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd"
OUTPUT="E1_CHT1 "
cpu=2

echo "Starting CANTS Experiment: " $i
time mpirun -n $cpu --oversubscribe python3 ./colonies.py --data_dir $dataFolder --data_files c172_file_1.csv --input_names $INPUT --output_names $OUTPUT --log_dir LOG_single_cantsbp --out_dir OUT_single_cantsbp --living_time 1000  --term_log_level INFO --log_file_name cants_trial_$f_name_$i --file_log_level INFO --col_log_level INFO -nrm minmax --num_col 1
