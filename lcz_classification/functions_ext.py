
# ALL FUNCTIONS IN THIS SCRIPT ARE ADOPTED FROM https://github.com/gisgeolab/LCZ-ODC/blob/Processing-Notebooks/functions.py

import pandas as pd
import numpy as np
import random
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import random
import numpy as np
import plotly.express as px
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

#------------------------------------------------#
# Function used in 6 - Validation.ipynb

def lcz_generator_accuracy(confusion_matrix_path, legend):
    
    confusion_matrix_df = pd.read_csv(confusion_matrix_path)
    confusion_matrix_df.set_index('Unnamed: 0', inplace = True)

    confusion_matrix_df = confusion_matrix_df.iloc[[1,2,4,5,7,10,11,13,14,15,16], [1,2,4,5,7,10,11,13,14,15,16]]
    confusion_matrix_df = confusion_matrix_df.rename(index = {'11': '101', '12': '102', '14': '104', '15': '105', '16': '106', '17': '107'})
    confusion_matrix_df = confusion_matrix_df.rename(columns = {'11': '101', '12': '102', '14': '104', '15': '105', '16': '106', '17': '107'})

    confusion_matrix = confusion_matrix_df.to_numpy()
    true_pos = np.diag(confusion_matrix)
    false_pos = np.sum(confusion_matrix, axis = 0) - true_pos
    false_neg = np.sum(confusion_matrix, axis = 1) - true_pos
    support = np.sum(confusion_matrix, axis = 1)

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1_score = 2*((precision*recall)/(precision+recall))

    fig = px.imshow(confusion_matrix_df, text_auto=True)

    # Update x and y ticks
    fig.update_xaxes(title_text = 'Classified', tickvals = list(legend.keys()), ticktext = [legend[key][0] for key in legend])
    fig.update_yaxes(title_text = 'Reference', tickvals = list(legend.keys()), ticktext = [legend[key][0] for key in legend])

    # Update heatmap size
    fig.update_layout(width = 800, height = 600, title = 'Confusion matrix')
    fig.show()

    data = {
        'precision': np.round(precision,2),
        'recall': np.round(recall,2),
        'f1-score': np.round(f1_score,2),
        'support': support.astype(int)
    }
    index_list = [legend[key][0] for key in legend]
    report = pd.DataFrame(data)
    report.index = index_list

    return report


#------------------------------------------------#
# Function used in 6 - Validation.ipynb



def random_sampling(raster1_path, raster2_path, raster3_path, output_raster1_path, output_raster2_path, output_raster3_path, pixels_number, classes_to_sample):

    # Number of pixels to extract from each class
    pixels_per_class = int(pixels_number / len(classes_to_sample))

    # Open the input rasters
    with rasterio.open(raster1_path) as src1, rasterio.open(raster2_path) as src2, rasterio.open(raster3_path) as src3:
        # Get raster metadata
        profile = src1.profile  # Use the profile of the first raster

        # Create output raster files
        with rasterio.open(output_raster1_path, 'w', **profile) as dst1, \
             rasterio.open(output_raster2_path, 'w', **profile) as dst2, \
             rasterio.open(output_raster3_path, 'w', **profile) as dst3:

            # Loop through each class
            for class_val in classes_to_sample:
                # Find pixels with the current class in raster1
                class_pixels_raster1 = np.where(src1.read(1) == class_val)
                num_pixels = len(class_pixels_raster1[0])

                if num_pixels <= pixels_per_class:
                    # If there are fewer pixels than required, take all of them
                    sampled_indices = list(range(num_pixels))
                else:
                    # Randomly sample 'pixels_per_class' indices
                    sampled_indices = random.sample(range(num_pixels), pixels_per_class)

                # Extract and write sampled pixels to the output rasters
                for index in sampled_indices:
                    row, col = class_pixels_raster1[0][index], class_pixels_raster1[1][index]
                    window = Window(col, row, 1, 1)  # Create a window around the pixel
                    sampled_pixel_raster1 = src1.read(window=window)
                    sampled_pixel_raster2 = src2.read(window=window)
                    sampled_pixel_raster3 = src3.read(window=window)
                    dst1.write(sampled_pixel_raster1, window=window)
                    dst2.write(sampled_pixel_raster2, window=window)
                    dst3.write(sampled_pixel_raster3, window=window)

    print("Sampled pixels exported to", output_raster1_path, ",", output_raster2_path, "and", output_raster3_path)


#------------------------------------------------#
# Function used in 6 - Validation.ipynb


def stratified_sampling(raster1_path, raster2_path, raster3_path, output_raster1_path, output_raster2_path, output_raster3_path, pixels_number, classes_to_sample):

    with rasterio.open(raster1_path) as src1, rasterio.open(raster2_path) as src2, rasterio.open(raster3_path) as src3:
        # Get raster metadata
        profile = src1.profile  # Use the profile of the first raster

        # Read the data as numpy arrays
        raster1_data = src1.read(1)
        raster2_data = src2.read(1)
        raster3_data = src3.read(1)

    # N_h: size of the population for h-th stratum
    valid_values_raster1 = raster1_data[~np.isnan(raster1_data)]
    total_pixels_raster1 = len(valid_values_raster1)
    valid_values_raster2 = raster2_data[~np.isnan(raster2_data)]
    total_pixels_raster2 = len(valid_values_raster2)
    valid_values_raster3 = raster3_data[~np.isnan(raster3_data)]
    total_pixels_raster3 = len(valid_values_raster3)

    unique_values_raster1, unique_counts_raster1 = np.unique(valid_values_raster1, return_counts=True)
    unique_values_raster2, unique_counts_raster2 = np.unique(valid_values_raster2, return_counts=True)
    unique_values_raster3, unique_counts_raster3 = np.unique(valid_values_raster3, return_counts=True)

    Nh_raster1 = dict(zip(unique_values_raster1.astype(int), unique_counts_raster1))
    Nh_raster2 = dict(zip(unique_values_raster2.astype(int), unique_counts_raster2))
    Nh_raster3 = dict(zip(unique_values_raster3.astype(int), unique_counts_raster3))

    # n: size of the entire sample
    n = pixels_number

    # N: size of the entire population
    N = sum(unique_counts_raster1)
    #N = sum(value_count_dict_raster1.values())

    # n_h: size of the sample for h-th stratum
    nh_raster1 = {key: value * n // N for key, value in Nh_raster1.items()}
    nh_raster2 = {key: value * n // N for key, value in Nh_raster2.items()}
    nh_raster3 = {key: value * n // N for key, value in Nh_raster3.items()}

    # Number of pixels to extract from each class
    pixels_per_class1 = nh_raster1.values()
    pixels_per_class2 = nh_raster2.values()
    pixels_per_class3 = nh_raster3.values()

    # Open the input rasters
    with rasterio.open(raster1_path) as src1, rasterio.open(raster2_path) as src2, rasterio.open(raster3_path) as src3:
        # Get raster metadata
        profile = src1.profile  # Use the profile of the first raster

        # Create output raster files
        with rasterio.open(output_raster1_path, 'w', **profile) as dst1, \
             rasterio.open(output_raster2_path, 'w', **profile) as dst2, \
             rasterio.open(output_raster3_path, 'w', **profile) as dst3:

            # Loop through each class
            for class_val, pixels1, pixels2, pixels3 in zip(classes_to_sample, pixels_per_class1, pixels_per_class2, pixels_per_class3):
                # Find pixels with the current class in raster1
                class_pixels_raster1 = np.where(src1.read(1) == class_val)
                num_pixels = len(class_pixels_raster1[0])

                if num_pixels <= pixels1:
                    # If there are fewer pixels than required, take all of them
                    sampled_indices = list(range(num_pixels))
                else:
                    # Randomly sample 'pixels1' indices
                    sampled_indices = random.sample(range(num_pixels), pixels1)

                # Extract and write sampled pixels to the output rasters
                for index in sampled_indices:
                    row, col = class_pixels_raster1[0][index], class_pixels_raster1[1][index]
                    window = Window(col, row, 1, 1)  # Create a window around the pixel
                    sampled_pixel_raster1 = src1.read(window=window)
                    sampled_pixel_raster2 = src2.read(window=window)
                    sampled_pixel_raster3 = src3.read(window=window)
                    dst1.write(sampled_pixel_raster1, window=window)
                    dst2.write(sampled_pixel_raster2, window=window)
                    dst3.write(sampled_pixel_raster3, window=window)

    print("Sampled pixels exported to", output_raster1_path, ",", output_raster2_path, "and", output_raster3_path)


#------------------------------------------------#
# Function used in 6 - Validation.ipynb

def inter_comparison(raster_path, ref_raster_path, legend):
    
    with rasterio.open(ref_raster_path) as src:
        mappa = src.read()
        modified_mappa = mappa.copy()
        modified_mappa=modified_mappa.astype(np.float64)
        modified_mappa[modified_mappa < 0] = np.nan
        unique_values_mappa = np.unique(modified_mappa)

    with rasterio.open(raster_path) as src:
        test = src.read()
        modified_test = test.copy()
        modified_test=modified_test.astype(np.float64)
        modified_test[modified_test < 0] = np.nan
        unique_values_test = np.unique(modified_test)

    # Retrieve the indices where testing_samples is greater than 0
    x, y, z = np.where(modified_test>0)

    # Select the testing samples that are greater than 0
    test_samples = modified_test[modified_test > 0]
    # Extract the corresponding classified samples from classified_image using the indices
    classified_samples = modified_mappa[x, y, z]

    # Retrieve the indices where classified_samples are greater than 0 and not NaN
    a = np.where((classified_samples > 0) & (~np.isnan(classified_samples)))

    # Filter the test_samples and classified_samples based on the indices obtained
    test_samples = test_samples[a]
    classified_samples = classified_samples[a]

    print('Test samples shape:', test_samples.shape)
    print('Classified samples shape:', classified_samples.shape)


    # Calculate the accuracy of the best model on the test set
    accuracy = accuracy_score(test_samples, classified_samples)
    print(f"Accuracy: {accuracy:.3f}")

    accuracy = accuracy_score(test_samples, classified_samples)
    confusion = confusion_matrix(test_samples, classified_samples)
    report = classification_report(test_samples, classified_samples)

    fig = px.imshow(confusion, text_auto=True)

    # Update x and y ticks
    fig.update_xaxes(title_text = 'PRISMA LCZ map', tickvals = list(range(len(legend.keys()))), ticktext = [legend[key][0] for key in legend])
    fig.update_yaxes(title_text = 'LCZ Generator map', tickvals = list(range(len(legend.keys()))), ticktext = [legend[key][0] for key in legend])

    # Update heatmap size
    fig.update_layout(width = 800, height = 600, title = 'Confusion matrix')

    fig.show()
    
    # Convert the report to a pandas df
    report1 = report.strip().split('\n')
    report_lists = [line.split() for line in report1]
    report_df = pd.DataFrame(report_lists[2:13])
    report_df.columns = ['LCZ', 'precision', 'recall', 'f1-score', 'support']
    report_df.index = [legend[key][0] for key in legend.keys()]
    report_df = report_df.iloc[:, 1:]
    
    return accuracy, confusion, report, report_df