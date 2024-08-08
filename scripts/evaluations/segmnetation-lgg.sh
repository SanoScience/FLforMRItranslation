#!/bin/bash -l


## Nazwa zlecenia
#SBATCH -J "SegEval"
# Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=1
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem-per-cpu=50GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=00:30:00
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgfmri2-gpu-a100
## Specyfikacja partycji
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
## Plik ze standardowym wyjściem
#SBATCH --output="logs/evaluations/segmentation/from_flair/lgg_noWdice.out"


directories=("lgg")

for dir in "${directories[@]}"; do
    srun $PLG_GROUPS_STORAGE/plggflmri/anaconda3/bin/python $HOME/repos/FLforMRItranslation/evaluate.py /net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/$dir/test\
     /net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/trained_models/model-lgg-MSE_DSSIM-ep20-FLAIRMASK-lr0.0001-2024-08-01-16h/best_model.pth 1
done