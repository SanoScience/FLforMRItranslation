#!/bin/bash -l

## Nazwa zlecenia
#SBATCH -J FedBadam
# Liczba alokowanych węzłów
#SBATCH -n 7
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
##SBATCH --mem=100GB
#SBATCH --mem-per-cpu=10GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=04:00:00
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgfmri2-gpu-a100
## Specyfikacja partycji
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gpus-per-task=1
## Plik ze standardowym wyjściem
#SBATCH --output="logs/training/t1_to_t2/FL/FedBadam/global.out"
#SBATCH --error="logs/training/t1_to_t2/FL/FedBadam/global.out"

DIR_NAME=logs/training/t1_to_t2/FL/FedBadam
echo $DIR_NAME

clients=("hgg_125" "oasis" "lgg" "ucsf_150" "hcp_mgh_masks" "hcp_wu_minn")

PORT=8088

## All the sruns are launched on the same node.
## --exclusive ensures that when the server starts there is room for the clients 
srun --ntasks=1 --cpus-per-task=1 \
    --output="./$DIR_NAME/server.out" --error="./$DIR_NAME/server_logs.out" \
    $PLG_GROUPS_STORAGE/plggflmri/anaconda3/bin/python -m \
    exe.trainings.run_server $PORT fedbadam &


sleep 300

for client in "${clients[@]}"; do
    srun --ntasks=1 --cpus-per-task=1 \
    --output="./$DIR_NAME/$client.out" --error="./$DIR_NAME/error_$client.out" \
    $PLG_GROUPS_STORAGE/plggflmri/anaconda3/bin/python -m exe.trainings.run_client_train \
    /net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/$client $client $PORT fedbadam &
done
wait