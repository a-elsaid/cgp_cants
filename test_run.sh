f_name="withBP"
activation="tanh"
#dataFolder="/Users/a.e./Dropbox/ANTS_CANTS/transformer_data/ETDataset/ETT-small"
#dataFolder="./datasets/2019_ngafid_transfer"
#INPUT="AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd"
#OUTPUT="E1_CHT1 "
#file_names="c172_file_1.csv"
#dataFolder="./datasets/iris"
#file_names="iris.csv"
#INPUT="sepal.length sepal.width petal.length petal.width"
#OUTPUT="variety"

dataFolder="./datasets/arrow"
file_names="arrowhead.csv"
INPUT="att1 att2 att3 att4 att5 att6 att7 att8 att9 att10 att11 att12 att13 att14 att15 att16 att17 att18 att19 att20 att21 att22 att23 att24 att25 att26 att27 att28 att29 att30 att31 att32 att33 att34 att35 att36 att37 att38 att39 att40 att41 att42 att43 att44 att45 att46 att47 att48 att49 att50 att51 att52 att53 att54 att55 att56 att57 att58 att59 att60 att61 att62 att63 att64 att65 att66 att67 att68 att69 att70 att71 att72 att73 att74 att75 att76 att77 att78 att79 att80 att81 att82 att83 att84 att85 att86 att87 att88 att89 att90 att91 att92 att93 att94 att95 att96 att97 att98 att99 att100 att101 att102 att103 att104 att105 att106 att107 att108 att109 att110 att111 att112 att113 att114 att115 att116 att117 att118 att119 att120 att121 att122 att123 att124 att125 att126 att127 att128 att129 att130 att131 att132 att133 att134 att135 att136 att137 att138 att139 att140 att141 att142 att143 att144 att145 att146 att147 att148 att149 att150 att151 att152 att153 att154 att155 att156 att157 att158 att159 att160 att161 att162 att163 att164 att165 att166 att167 att168 att169 att170 att171 att172 att173 att174 att175 att176 att177 att178 att179 att180 att181 att182 att183 att184 att185 att186 att187 att188 att189 att190 att191 att192 att193 att194 att195 att196 att197 att198 att199 att200 att201 att202 att203 att204 att205 att206 att207 att208 att209 att210 att211 att212 att213 att214 att215 att216 att217 att218 att219 att220 att221 att222 att223 att224 att225 att226 att227 att228 att229 att230 att231 att232 att233 att234 att235 att236 att237 att238 att239 att240 att241 att242 att243 att244 att245 att246 att247 att248 att249 att250 att251"
OUTPUT="target"
cpu=9

echo "Starting CANTS Experiment: " $i
time mpirun -n $cpu --oversubscribe python3 ./colonies.py --data_dir $dataFolder --data_files $file_names --input_names $INPUT --output_names $OUTPUT --log_dir LOG_single_cantsbp --out_dir OUT_single_cantsbp --living_time 1000  --term_log_level INFO --log_file_name cants_trial_$f_name_$i --file_log_level INFO --col_log_level INFO -nrm minmax --num_col 8
