#/bin/bash

workflow='GenerateData'

mkdir ../Workflows/"$workflow"

cp inputs.txt ../Workflows/"$workflow"
cp submission.sh ../Workflows/"$workflow"
cp train.py ../Workflows/"$workflow"
cp -r thermal_model ../Workflows/"$workflow"
cp -r dnn_model ../Workflows/"$workflow"

Xi_t=$(sed -n '4p' inputs.txt)
C_t=$(sed -n '5p' inputs.txt)
k_t=$(sed -n '6p' inputs.txt)
t_t=$(sed -n '7p' inputs.txt)
R_12=$(sed -n '8p' inputs.txt)
C_i=$(sed -n '9p' inputs.txt)
k_i=$(sed -n '10p' inputs.txt)
t_i=$(sed -n '11p' inputs.txt)
R_23=$(sed -n '12p' inputs.txt)
C_s=$(sed -n '13p' inputs.txt)
k_s=$(sed -n '14p' inputs.txt)
r_pump=$(sed -n '15p' inputs.txt)
r_probe=$(sed -n '16p' inputs.txt)
t_delay=$(sed -n '17p' inputs.txt)
t_res=$(sed -n '18p' inputs.txt)

SNR=$(sed -n '21p' inputs.txt)
maxNoise=$(sed -n '22p' inputs.txt)
pct_pinkNoise=$(sed -n '23p' inputs.txt)

N_data=$(sed -n '26p' inputs.txt)
pct_dev=$(sed -n '27p' inputs.txt)
pct_test=$(sed -n '28p' inputs.txt)

cd ../Workflows/"$workflow"

sed -i -e "3i $Xi_t" -e "3i $C_t" -e "3i $k_t" -e "3i $t_t" -e "3i $R_12" -e "3i $C_i" -e "3i $k_i" -e "3i $t_i" -e "3i $R_23" -e "3i $C_s" -e "3i $k_s" -e \
"3i $r_pump" -e "3i $r_probe" \
-e "3i $t_delay" -e "3i $t_res" \
-e "4i $SNR" -e "4i $maxNoise" -e "4i $pct_pinkNoise" \
-e "5i $N_data" -e "5i $pct_dev" -e "5i $pct_test" submission.sh

sed -i -e 's/\r$//' submission.sh