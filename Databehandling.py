import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nibabel.processing import resample_from_to

# patienter och veckor
patient = 'Patient-086'
weeks = ['week-173-1', 'week-173-2']
base_dir = r'C:\Users\xhaaab\Downloads\Imaging-v202211 (1)\Imaging'
patient_path = os.path.join(base_dir, patient)

# Masscentrum slice
slice_z = 103

# tumörregionen
tumor_slices = {}

for week in weeks:
    seg_path = os.path.join(patient_path, week, 'DeepBraTumIA-segmentation', 'atlas', 'segmentation', 'seg_mask.nii.gz')
    t2_path = os.path.join(patient_path, week, 'DeepBraTumIA-segmentation', 'atlas', 'skull_strip', 't2_skull_strip.nii.gz')

    if not os.path.exists(seg_path) or not os.path.exists(t2_path):
        print(f"❌ Missing files for {week}")
        continue

    # laddar upp segmenteringen och mr bilden
    seg_img = nib.load(seg_path)
    seg_data = seg_img.get_fdata()

    mr_img = nib.load(t2_path)
    mr_resampled = resample_from_to(mr_img, seg_img)  # länkar t2 till segmenteringen. de koordinater segmenteringen är i fås ut från t2
    mr_data = mr_resampled.get_fdata()

    # Få ut slice z
    if slice_z >= seg_data.shape[2]:
        print(f"⚠ Slice {slice_z} out of bounds for {week} (max z: {seg_data.shape[2]-1})")
        continue

    seg_slice = seg_data[:, :, slice_z]
    mr_slice = mr_data[:, :, slice_z]

    # Tar ut den segmenterade tumörområdet från T2 intensiteter
    tumor_mask = seg_slice > 0
    tumor_intensities = np.zeros_like(mr_slice)
    tumor_intensities[tumor_mask] = mr_slice[tumor_mask]

    #intensitetgitter
    tumor_slices[week] = tumor_intensities

    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(tumor_intensities.T, cmap='hot', origin='lower')
    plt.title(f"Tumor intensity - {week} (slice {slice_z})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    output_dir = os.path.join(r"C:\Users\xhaaab\Downloads\Finaldata\Patient86", week)
    output_csv_path = os.path.join(output_dir, f"tumor_intensity_slice_{slice_z}.csv")
#diskretiseran i 12 delar och normaliserar.
    max_val = np.max(tumor_intensities)
    if max_val > 0:
        normalized = tumor_intensities / max_val
        discretized = np.floor(normalized * 12) / 12
        discretized[discretized < 0.2] = 0.0
    else:
        discretized = tumor_intensities  # leave untouched if max is 0

    np.savetxt(output_csv_path, discretized, delimiter=",", fmt="%.1f")
    print(f"Sparat tumör i: {output_csv_path}")
