#!/bin/bash -l


## Nazwa zlecenia
#SBATCH -J "3dDICE"
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
#SBATCH --output="logs/evaluations/segmentation/from_flair/tested_hgg_125/DL_smooth1/3d_dice/oasis.out"

tested_model_dir=model-oasis-MSE_DSSIM-ep50-T1FLAIR-lr0.001-2024-02-22-10h

srun $PLG_GROUPS_STORAGE/plggflmri/anaconda3/bin/python $HOME/repos/FLforMRItranslation/compute_dice3d.py /net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/hgg_125/test/mask\
    /net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/trained_models/$tested_model_dir/preds/hgg_125/segmenatation/
