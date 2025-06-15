import sys

from src.utils.files_operations import TransformNIIDataToNumpySlices

if __name__ == '__main__':
    target_root_dir = sys.argv[1]
    current_data_dir = sys.argv[2]
    if len(sys.argv) > 3:
        n_patients = int(sys.argv[3])
    else:
        n_patients = -1

    transpose_order = (2, 0, 1)
    transformer = TransformNIIDataToNumpySlices(target_root_dir, current_data_dir, transpose_order, target_zero_ratio=0.8, leave_patient_name=False)
    transformer.create_train_val_test_sets("T_bet/T1_bet_reg.nii.gz", "T_bet/T2_bet_reg.nii.gz", "flair_bet/*FLAIR_bet.nii.gz", n_patients=n_patients)
