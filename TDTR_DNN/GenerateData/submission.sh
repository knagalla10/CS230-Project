#!/bin/bash




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

#SBATCH --cpus-per-task=20

module load matlab/r2024b
matlab -batch "GenerateData($SNR,$maxNoise,$pct_pinkNoise,$N_data,$pct_dev,$pct_test,$input)"
EOT