#!/bin/bash

Xi_t=(20 30)
C_t=(2.3 2.6)
k_t=(170 250)
t_t=(70 100)
R_12=(1 20)
C_i=(2.2 2.6)
k_i=(10 100)
t_i=(400 600)
R_23=(1 20)
C_s=(1.4 1.8)
k_s=(140 160)
r_pump=(10 14)
r_probe=(6 10)
t_delay=(-1.75 1.75)
t_res=(13 17)

SNR=5000
maxNoise=0.02
pct_pinkNoise=50

N_data=100000
pct_dev=10
pct_test=10

props="${Xi_t[@]} ${C_t[@]} ${k_t[@]} ${t_t[@]} ${R_12[@]} ${C_i[@]} ${k_i[@]} ${t_i[@]} ${R_23[@]} ${C_s[@]} ${k_s[@]} ${r_pump[@]} ${r_probe[@]} ${t_delay[@]} ${t_res[@]}"

input=()
for num in "${props[@]}";
do
    input+="$num,"
done
input="{${input%,}}"

cd thermal_model

sbatch << EOT
#!/bin/bash
#SBATCH --job-name=generate-data
#SBATCH --time=2-00:00:00
#SBATCH -o job-%J.txt
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=knagalla@stanford.edu

#SBATCH --cpus-per-task=40

module load matlab/r2024b
matlab -batch "GenerateData($SNR,$maxNoise,$pct_pinkNoise,$N_data,$pct_dev,$pct_test,$input)"
EOT