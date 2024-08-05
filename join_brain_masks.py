from src.utils import files_operations as fop
import sys


if __name__ == '__main__':
    # target_root_dir = sys.argv[1]
    # current_data_dir = sys.argv[2]
    if len(sys.argv) > 2:
        processed_dir_name = sys.argv[1]
        mask_dir_name = sys.argv[2]
    else:
        processed_dir_name = "C:\\Users\\JanFiszer\\data\\mri\\hgg_valid_t1"
        mask_dir_name = "C:\\Users\\JanFiszer\\data\\mri\\HGG"

    # patient_slices = fop.get_brains_slices_info(dir_name)
    fop.create_segmentation_mask_dir(processed_dir_name, mask_dir_name, transpose_order=(2, 0, 1), only_with_glioma=True)
