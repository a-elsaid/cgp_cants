f_name="withBP"
activation="tanh"
#dataFolder="/Users/a.e./Dropbox/ANTS_CANTS/datasets/2020_engie_wind"
#dataFolder="/Users/a.e./Dropbox/ANTS_CANTS/datasets/2019_ngafid_transfer"
dataFolder="/Users/a.e./Dropbox/ANTS_CANTS/datasets/2018_coal"
#INPUT="AltAGL AltB AltGPS AltMSL BaroA E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd E1_CHT1"
#OUTPUT="E1_CHT1"
INPUT="Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int"
OUTPUT="Supp_Fuel_Flow"
#file_names="turbine_R80721_2013-2016_1.csv"
#file_names="c172_file_1.csv"
file_names="burner_0.csv"
cpu=5

for i in {1..9}; do
    echo "Starting CANTS Experiment: " $i
    echo $dataFolder
    time mpirun -n $cpu --oversubscribe python3 ./src/colonies.py --data_dir $dataFolder --data_files $file_names --input_names $INPUT --output_names $OUTPUT --log_dir LOG_single_cantsbp --out_dir OUT_single_cantsbp --living_time 75  --term_log_level INFO --log_file_name cants_trial_$f_name_$i --file_log_level INFO --col_log_level INFO -nrm minmax --comm_interval 2 #--num_col 2
    exit
done
