import sys

from src.utils.files_operations import TransformNIIDataToNumpySlices

"""Script to transform NIfTI MRI datasets into training slices.
Converts 3D MRI volumes into 2D slices for federated learning."""

if __name__ == '__main__':
    # Get directory paths from command line arguments
    target_root_dir = sys.argv[1]  # Directory where processed data will be saved
    current_data_dir = sys.argv[2]  # Source directory containing NIfTI files

    # Optional: number of patients to process (-1 for all)
    if len(sys.argv) > 3:
        n_patients = int(sys.argv[3])
    else:
        n_patients = -1

    # Configure axes order for transposing 3D volumes
    transpose_order = (2, 0, 1)  # Rearrange to slice-height-width format
    
    # Initialize transformer with 80% minimum non-zero content requirement
    transformer = TransformNIIDataToNumpySlices(
        target_root_dir, current_data_dir, transpose_order, 
        target_zero_ratio=0.8,  # Skip slices with >80% zero values
        leave_patient_name=False  # Use generic patient IDs
    )
    
    # Process T1, T2 and FLAIR modalities
    transformer.create_train_val_test_sets(
        "T_bet/T1_bet_reg.nii.gz",  # Path pattern for T1 images
        "T_bet/T2_bet_reg.nii.gz",  # Path pattern for T2 images 
        "flair_bet/*FLAIR_bet.nii.gz",  # Path pattern for FLAIR images
        n_patients=n_patients
    )
