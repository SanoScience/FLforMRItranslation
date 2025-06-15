import sys

from src.utils.files_operations import TransformNIIDataToNumpySlices

if __name__ == '__main__':
    # target_root_dir = sys.argv[1]
    # current_data_dir = sys.argv[2]
    if len(sys.argv) > 3:
        n_patients = int(sys.argv[3])
    else:
        n_patients = -1
    current_data_dir = "C:\\Users\\JanFiszer\\data\\mri\\Oasis_masks_with_flair"
    target_root_dir = "C:\\Users\\JanFiszer\\data\\mri\\oasis_flair"
    n_patients = 5

    # C:\Users\JanFiszer\data\HCP_Wu-Minn\114318_3T_Structural_preproc\114318\T1w
    transpose_order = (2, 0, 1)
    transformer = TransformNIIDataToNumpySlices(target_root_dir, current_data_dir, transpose_order, target_zero_ratio=0.8, leave_patient_name=False)
    # transformer.create_train_val_test_sets("T1_bet/T1_bet.nii.gz", "T2_bet/T2_bet_reg.nii.gz", n_patients=n_patients, seed=42)
    # transformer.create_train_val_test_sets("*T1.nii.gz", "*T2.nii.gz", "*FLAIR.nii.gz", n_patients=n_patients)  # for ucsf  transpose_order = (2, 0, 1)
    transformer.create_train_val_test_sets("T_bet/T1_bet_reg.nii.gz", "T_bet/T2_bet_reg.nii.gz", "flair_bet/*FLAIR_bet.nii.gz", n_patients=n_patients)
