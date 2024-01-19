#!/bin/bash -l

## Nazwa zlecenia
#SBATCH -J Mem
# Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=6
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem=100GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=01:00:00
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgplgflmri-gpu-a100
## Specyfikacja partycji
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:6
## Plik ze standardowym wyjściem
#SBATCH --output="logs/t1_to_t2/FedAvg/global.out"
#SBATCH --error="logs/t1_to_t2/FedAvg/global.out"

DIR_NAME=logs/t1_to_t2/FedAvg
echo $DIR_NAME

clients=("hgg_50" "oasis_125" "lgg" "hcp_mgh_masks" "hcp_wu_minn")

PORT=8087

## All the sruns are launched on the same node.
## --exclusive ensures that when the server starts there is room for the clients 
srun --ntasks=1 --gpus-per-task=1 --exclusive --mem-per-cpu=15GB \
    --output="./$DIR_NAME/server.out" --error="./$DIR_NAME/server_logs.out" \
    $PLG_GROUPS_STORAGE/plggflmri/anaconda3/bin/python \
    $HOME/repos/FLforMRItranslation/run_server.py $PORT fedavg &


sleep 700

for client in "${clients[@]}"; do
    srun --ntasks=1 --exclusive --gpus-per-task=1 \
    --output="./$DIR_NAME/$client.out" --error="./$DIR_NAME/error_$client.out" \
    $PLG_GROUPS_STORAGE/plggflmri/anaconda3/bin/python $HOME/repos/FLforMRItranslation/run_client_train.py \
    /net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/$client $client $PORT fedavg &
done
wait