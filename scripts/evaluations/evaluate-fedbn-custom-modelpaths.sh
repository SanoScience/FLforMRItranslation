#!/bin/bash -l


## Nazwa zlecenia
#SBATCH -J "EvalBN"
# Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=1
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem-per-cpu=50GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=00:20:00
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgplgflmri-gpu-a100
## Specyfikacja partycji
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
## Plik ze standardowym wyjściem
#SBATCH --output="logs/evaluations/t2_to_t1/FL/FedBN.out"

data_directories=("ucsf_150" "hgg_125" "lgg" "hcp_mgh_masks" "hcp_wu_minn" "oasis")
model_directories=("ucsf_150" "hgg_125" "lgg" "hcp_mgh_masks" "hcp_wu_minn" "oasis")


for ((i = 0; i < ${#data_directories[@]}; i++)); do
    data_dir="${data_directories[i]}"
    model_dir="${model_directories[i]}"

    srun $PLG_GROUPS_STORAGE/plggflmri/anaconda3/bin/python $HOME/repos/FLforMRItranslation/evaluate.py /net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/$data_dir/test \
    "/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/trained_models/model-fedbn-MSE_DSSIM-T2T1-lr0.001-rd32-ep4-2024-03-15/FedBN(batch_norm=NormalizationType.BN)_client_$model_dir/model-rd20.pth" 16
done