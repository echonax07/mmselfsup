import os
import numpy as np
import xarray as xr
import random
from tqdm import tqdm
def calculate_statistics(folder_path, sample_size=40):
    # Initialize variables to store sum and count for each channel
    nersc_sar_primary_sum = 0
    nersc_sar_primary_count = 0
    nersc_sar_secondary_sum = 0
    nersc_sar_secondary_count = 0
    sar_incidenceangle_sum = 0
    sar_incidenceangle_count = 0
    files = [f for f in os.listdir(folder_path) if f.endswith(".nc")]
     # Randomly select 'sample_size' files from the list
    selected_files = random.sample(files, min(len(files), sample_size))
    # Iterate through all files in the folder
    for filename in tqdm(selected_files, desc='Calculating Mean'):
            with xr.open_dataset(os.path.join(folder_path,filename), engine='h5netcdf') as nc_file:
                # Extract data for each channel
                nersc_sar_primary_data = nc_file.variables['nersc_sar_primary'].values
                nersc_sar_primary_data = nersc_sar_primary_data.flatten()[~np.isnan(nersc_sar_primary_data.flatten())]
                nersc_sar_secondary_data = nc_file.variables['nersc_sar_secondary'].values
                nersc_sar_secondary_data = nersc_sar_secondary_data.flatten()[~np.isnan(nersc_sar_secondary_data.flatten())]
                sar_incidenceangle_data = nc_file.variables['sar_grid_incidenceangle'].values
                sar_incidenceangle_data = sar_incidenceangle_data.flatten()[~np.isnan(sar_incidenceangle_data.flatten())]

                
                # Update sum and count for each channel
                nersc_sar_primary_sum += np.sum(nersc_sar_primary_data)
                nersc_sar_primary_count += nersc_sar_primary_data.size
                nersc_sar_secondary_sum += np.sum(nersc_sar_secondary_data)
                nersc_sar_secondary_count += nersc_sar_secondary_data.size
                sar_incidenceangle_sum += np.sum(sar_incidenceangle_data)
                sar_incidenceangle_count += sar_incidenceangle_data.size

    # Calculate mean for each channel
    nersc_sar_primary_mean = nersc_sar_primary_sum / nersc_sar_primary_count
    nersc_sar_secondary_mean = nersc_sar_secondary_sum / nersc_sar_secondary_count
    sar_incidenceangle_mean = sar_incidenceangle_sum / sar_incidenceangle_count

    # Re-iterate to calculate standard deviation for each channel
    nersc_sar_primary_sum_of_squares = 0
    nersc_sar_secondary_sum_of_squares = 0
    sar_incidenceangle_sum_of_squares = 0
    for filename in tqdm(selected_files, desc='Calculating Std'):
            with xr.open_dataset(os.path.join(folder_path,filename), engine='h5netcdf') as nc_file:
                nersc_sar_primary_data = nc_file.variables['nersc_sar_primary'].values
                nersc_sar_primary_data = nersc_sar_primary_data.flatten()[~np.isnan(nersc_sar_primary_data.flatten())]
                nersc_sar_secondary_data = nc_file.variables['nersc_sar_secondary'].values
                nersc_sar_secondary_data = nersc_sar_secondary_data.flatten()[~np.isnan(nersc_sar_secondary_data.flatten())]
                sar_incidenceangle_data = nc_file.variables['sar_grid_incidenceangle'].values
                sar_incidenceangle_data = sar_incidenceangle_data.flatten()[~np.isnan(sar_incidenceangle_data.flatten())]

                nersc_sar_primary_sum_of_squares += np.sum((nersc_sar_primary_data - nersc_sar_primary_mean) ** 2)
                nersc_sar_secondary_sum_of_squares += np.sum((nersc_sar_secondary_data - nersc_sar_secondary_mean) ** 2)
                sar_incidenceangle_sum_of_squares += np.sum((sar_incidenceangle_data - sar_incidenceangle_mean) ** 2) 

    # Calculate standard deviation for each channel
    nersc_sar_primary_std = np.sqrt(nersc_sar_primary_sum_of_squares / nersc_sar_primary_count)
    nersc_sar_secondary_std = np.sqrt(nersc_sar_secondary_sum_of_squares / nersc_sar_secondary_count)
    sar_incidenceangle_std = np.sqrt(sar_incidenceangle_sum_of_squares / sar_incidenceangle_count)

    return (nersc_sar_primary_mean, nersc_sar_primary_std), (nersc_sar_secondary_mean, nersc_sar_secondary_std), \
           (sar_incidenceangle_mean, sar_incidenceangle_std)

def save_to_txt(means_primary, stds_primary, means_secondary, stds_secondary, means_incidenceangle, stds_incidenceangle ,output_file):
    with open(output_file, 'w') as file:
        file.write("Channel\tMean\tStandard Deviation\n")
        file.write("nersc_sar_primary\t{}\t{}\n".format(means_primary, stds_primary))
        file.write("nersc_sar_secondary\t{}\t{}\n".format(means_secondary, stds_secondary))
        file.write("sar_incidenceangle\t{}\t{}\n".format(means_incidenceangle, stds_incidenceangle))

# Example usage
folder_path = '/home/m32patel/projects/def-dclausi/AI4arctic/dataset/ai4arctic_raw_train_v3'
output_file = '/home/m32patel/projects/def-dclausi/AI4arctic/m32patel/mmselfsup/ai4arctic/statistics.txt'

statistics = calculate_statistics(folder_path, sample_size=533)

# Unpack the results
means_primary, stds_primary = statistics[0]
means_secondary, stds_secondary = statistics[1]
means_incidenceangle, stds_incidenceangle = statistics[2]

save_to_txt(means_primary, stds_primary, means_secondary, stds_secondary, means_incidenceangle, stds_incidenceangle ,output_file)
print("Statistics saved to", output_file)
