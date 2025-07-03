f_name="gp-cants"
activation="tanh"

declare DATA_DIRS
declare INPUT
declare OUTPUT
declare FILE_NAMES

DATA_DIRS[wind]="/Users/a.e./Dropbox/ANTS_CANTS/datasets/2020_engie_wind"
DATA_DIRS[c172]="/Users/a.e./Dropbox/ANTS_CANTS/datasets/2019_ngafid_transfer"
DATA_DIRS[burner]="/Users/a.e./Dropbox/ANTS_CANTS/datasets/2018_coal"

INPUT[wind]="AltAGL AltB AltGPS AltMSL BaroA E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd E1_CHT1"
INPUT[burner]="Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int"

OUTPUT[c172]="E1_CHT1"
OUTPUT[burner]="Supp_Fuel_Flow"

FILE_NAMES[wind]="turbine_R80721_2013-2016_1.csv"
FILE_NAMES[c172]="c172_file_1.csv"
FILE_NAMES[burner]="burner_0.csv"

cpu=9                  # this will create 1 Enviroment Process & cpu-1 Colonies

set=burner

for i in {1..9}; do
    echo "Starting CANTS Experiment: " $i
    time mpirun -n $cpu --oversubscribe python3 ./src/colonies.py \
    --data_dir ${DATA_DIRS[$set]} \
    --data_files ${FILE_NAMES[$set]} \
    --input_names ${INPUT[$set]} \
    --output_names ${OUTPUT[$set]} \
    --log_dir LOG\
    --out_dir OUT\
    --living_time 90  \
    --term_log_level INFO \
    --log_file_name cants_trial_$f_name_$i \
    --file_log_level INFO \
    --col_log_level INFO -nrm minmax \
    --use_bp --bp_epochs 9 \
    --loss_fun mse \
    --comm_interval 5
    exit
done
