#!/bin/bash -l


## Nazwa zlecenia
#SBATCH -J "EvalAvg"
# Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=1
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem-per-cpu=50GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=01:00:00
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgfmri2-gpu-a100
## Specyfikacja partycji
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
## Plik ze standardowym wyjściem
#SBATCH --output="logs/evaluations/flair_to_t2/fl/FedAvg.out"


directories=("ucsf_150" "hgg_125" "lgg" "oasis")

for dir in "${directories[@]}"; do
    srun $PLG_GROUPS_STORAGE/plggflmri/anaconda3/bin/python $HOME/repos/FLforMRItranslation/evaluate.py /net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/$dir/test\
     /net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/trained_models/model-fedavg-MSE_DSSIM-FLAIRT2-lr0.001-rd32-ep4-2024-04-06/round32.pth 16
done