#!/bin/bash -l


## Nazwa zlecenia
#SBATCH -J "t1t2hgg"
# Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=1
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem-per-cpu=150GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=01:30:00
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgfmri3-gpu-a100
## Specyfikacja partycjiW
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
## Plik ze standardowym wyjściem
#SBATCH --output="logs/training/t1_to_t2/centralized/hgg_125.out"
#SBATCH --error="logs/training/t1_to_t2/centralized/hgg_125.err"


srun $PLG_GROUPS_STORAGE/plggflmri/anaconda3/bin/python $HOME/repos/FLforMRItranslation/classical_train.py /net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/hgg_125
