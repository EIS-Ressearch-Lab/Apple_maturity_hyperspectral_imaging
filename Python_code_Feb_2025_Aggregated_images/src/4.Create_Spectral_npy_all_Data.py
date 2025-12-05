#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
print(torch.cuda.is_available())


from detecto import core, utils, visualize
from detecto.visualize import show_labeled_image, plot_prediction_grid
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import re
import os
import gc

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


# In[2]:


# load data_final to pickle
pickle_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Pickle_files/'

data_NZ2023_a_final2 = pd.read_pickle(f'{pickle_path}data_NZ2023_a_final2.pkl')
data_NZ2023_b_final2 = pd.read_pickle(f'{pickle_path}data_NZ2023_b_final2.pkl')
data_NZ2023_c_final2 = pd.read_pickle(f'{pickle_path}data_NZ2023_c_final2.pkl')
data_NZ2023_d_final2 = pd.read_pickle(f'{pickle_path}data_NZ2023_d_final2.pkl')
data_NZ2024_a_final2 = pd.read_pickle(f'{pickle_path}data_NZ2024_a_final2.pkl')
data_NZ2024_b_final2 = pd.read_pickle(f'{pickle_path}data_NZ2024_b_final2.pkl')
data_NZ2024_c_final2 = pd.read_pickle(f'{pickle_path}data_NZ2024_c_final2.pkl')
data_NZ2024_d_final2 = pd.read_pickle(f'{pickle_path}data_NZ2024_d_final2.pkl')
data_UK2024_a_final2 = pd.read_pickle(f'{pickle_path}data_UK2024_a_final2.pkl')
data_UK2024_b_final2 = pd.read_pickle(f'{pickle_path}data_UK2024_b_final2.pkl')
data_UK2024_c_final2 = pd.read_pickle(f'{pickle_path}data_UK2024_c_final2.pkl')
data_UK2024_d_final2 = pd.read_pickle(f'{pickle_path}data_UK2024_d_final2.pkl')


# In[3]:


print(data_NZ2023_a_final2.head())
print(data_NZ2024_b_final2.head())
print(data_UK2024_c_final2.head())


# In[4]:


# Spectral folder

# Split by '/' and remove last split in string

data_NZ2023_a_final2['Spectral_folder_a'] = data_NZ2023_a_final2['Image_folder_A'].str.split('/').str[:-1].str.join('/')
data_NZ2023_b_final2['Spectral_folder_b'] = data_NZ2023_b_final2['Image_folder_B'].str.split('/').str[:-1].str.join('/')
data_NZ2023_c_final2['Spectral_folder_c'] = data_NZ2023_c_final2['Image_folder_C'].str.split('/').str[:-1].str.join('/')
data_NZ2023_d_final2['Spectral_folder_d'] = data_NZ2023_d_final2['Image_folder_D'].str.split('/').str[:-1].str.join('/')
data_NZ2024_a_final2['Spectral_folder_a'] = data_NZ2024_a_final2['Image_folder_A'].str.split('/').str[:-1].str.join('/')
data_NZ2024_b_final2['Spectral_folder_b'] = data_NZ2024_b_final2['Image_folder_B'].str.split('/').str[:-1].str.join('/')
data_NZ2024_c_final2['Spectral_folder_c'] = data_NZ2024_c_final2['Image_folder_C'].str.split('/').str[:-1].str.join('/')
data_NZ2024_d_final2['Spectral_folder_d'] = data_NZ2024_d_final2['Image_folder_D'].str.split('/').str[:-1].str.join('/')
data_UK2024_a_final2['Spectral_folder_a'] = data_UK2024_a_final2['Image_folder_A'].str.split('/').str[:-1].str.join('/')
data_UK2024_b_final2['Spectral_folder_b'] = data_UK2024_b_final2['Image_folder_B'].str.split('/').str[:-1].str.join('/')
data_UK2024_c_final2['Spectral_folder_c'] = data_UK2024_c_final2['Image_folder_C'].str.split('/').str[:-1].str.join('/')
data_UK2024_d_final2['Spectral_folder_d'] = data_UK2024_d_final2['Image_folder_D'].str.split('/').str[:-1].str.join('/')


# In[5]:


print(data_NZ2023_a_final2['Spectral_folder_a'][0])
print(data_NZ2024_b_final2['Spectral_folder_b'][0])
print(data_UK2024_c_final2['Spectral_folder_c'][0])


# In[7]:


import numpy as np
import glob

# More memory efficient version of process_spectral_images
# def spectral_image_generator(dataframe, folder_path, side, height=512, width=512, aggregate_pixel_height=15, aggregate_pixel_width=15):
#     """
#     Generator function to process hyperspectral images by cropping to fit an exact grid and aggregating pixels.

#     Parameters:
#     - dataframe: Pandas DataFrame containing image metadata
#     - folder_path: Path to the folder containing spectral data
#     - side: Specifies which column to use for spectral folder ('a', 'b', etc.)
#     - height: Default image height (512)
#     - width: Default image width (512)
#     - aggregate_pixel_height: Number of rows in aggregated output (15)
#     - aggregate_pixel_width: Number of columns in aggregated output (15)

#     Yields:
#     - aggregated_array: Processed 15x15 aggregated image
#     - row: Corresponding metadata row from dataframe
#     """

#     for index, row in dataframe.iterrows():
#         print(f"Processing {index + 1}/{len(dataframe)}")

#         spectral_link = row[f'Spectral_folder_{side}']
#         dat_files = glob.glob(f"{folder_path}/{spectral_link}/results/*.dat")
        
#         if not dat_files:
#             print(f"No .dat file found in {spectral_link}. Skipping...")
#             continue

#         # Load spectral data
#         spectral_data = np.fromfile(dat_files[0], dtype=np.float32)
#         bands = len(spectral_data) // (height * width)
#         spectral_data = spectral_data.reshape(height, width, bands)

#         # Process each bounding box in "sorted_tensor"
#         for region in row["sorted_tensor"]:
#             x1, y1, x2, y2 = map(int, region)
#             image_segment = spectral_data[y1:y2, x1:x2]

#             h, w, c = image_segment.shape

#             # Compute pixels to remove for divisibility by aggregate pixel size
#             remove_h = h % aggregate_pixel_height
#             remove_w = w % aggregate_pixel_width

#             # Crop from edges to prioritize the center
#             top_crop = remove_h // 2
#             bottom_crop = remove_h - top_crop
#             left_crop = remove_w // 2
#             right_crop = remove_w - left_crop

#             # Apply cropping
#             image_segment = image_segment[top_crop:h-bottom_crop, left_crop:w-right_crop]

#             # Ensure dimensions are now divisible by aggregate pixel size
#             h, w, c = image_segment.shape
#             assert h % aggregate_pixel_height == 0 and w % aggregate_pixel_width == 0, "Cropping failed!"

#             # Compute block size
#             block_h, block_w = h // aggregate_pixel_height, w // aggregate_pixel_width

#             # Perform pixel aggregation
#             aggregated_array = np.zeros((aggregate_pixel_height, aggregate_pixel_width, c), dtype=np.float32)
#             for m in range(aggregate_pixel_height):
#                 for n in range(aggregate_pixel_width):
#                     y_start, y_end = m * block_h, (m + 1) * block_h
#                     x_start, x_end = n * block_w, (n + 1) * block_w
#                     aggregated_array[m, n] = np.mean(image_segment[y_start:y_end, x_start:x_end], axis=(0, 1))

#             # Free memory after processing
#             del image_segment
#             gc.collect()

#             yield aggregated_array, row  # Return results one by one


def process_spectral_images(dataframe, folder_path, side, height=512, width=512, aggregate_pixel_height=15, aggregate_pixel_width=15):
    """
    Process hyperspectral images by cropping to fit an exact grid and aggregating pixels.

    Parameters:
    - dataframe: Pandas DataFrame containing image metadata
    - side: Specifies which column to use for spectral folder ('a', 'b', etc.)
    - height: Default image height
    - width: Default image width
    - aggregate_pixel_height: Number of rows in aggregated output
    - aggregate_pixel_width: Number of columns in aggregated output

    Returns:
    - processed_images: List of 15x15 aggregated images
    - file_metadata: List of associated file metadata (rows from dataframe)
    """

    processed_images = []
    file_metadata = []

    for index, row in dataframe.iterrows():
        print(f"Processing {index + 1}/{len(dataframe)}")

        spectral_link = row[f'Spectral_folder_{side}']
        # print(spectral_link)
        # data_file_path = f"{folder_path}/{spectral_link}/results/*.dat"
        # print(data_file_path)
        dat_files = glob.glob(f"{folder_path}/{spectral_link}/results/*.dat")
        
        if not dat_files:
            print(f"No .dat file found in {spectral_link}. Skipping...")
            continue

        # Load spectral data
        spectral_data = np.fromfile(dat_files[0], dtype=np.float32)
        bands = len(spectral_data) // (height * width)
        spectral_data = spectral_data.reshape(height, bands, width)
        spectral_data = np.transpose(spectral_data, (2, 0, 1))  # Convert to (H, W, B)

        spectral_flipped = np.fliplr(spectral_data)


        # Process each bounding box in "sorted_tensor"
        for region in row["sorted_tensor"]:
            x1, y1, x2, y2 = map(int, region)
            image_segment = spectral_flipped[y1:y2, x1:x2]
            # image_segment = spectral_data[y1:y2, x1:x2]

            h, w, c = image_segment.shape

            # Compute pixels to remove for divisibility by aggregate pixel size
            remove_h = h % aggregate_pixel_height
            remove_w = w % aggregate_pixel_width

            # Crop from edges to prioritize the center
            top_crop = remove_h // 2
            bottom_crop = remove_h - top_crop
            left_crop = remove_w // 2
            right_crop = remove_w - left_crop

            # Apply cropping
            image_segment = image_segment[top_crop:h-bottom_crop, left_crop:w-right_crop]

            # Ensure dimensions are now divisible by aggregate pixel size
            h, w, c = image_segment.shape
            assert h % aggregate_pixel_height == 0 and w % aggregate_pixel_width == 0, "Cropping failed!"

            # Compute block size
            block_h, block_w = h // aggregate_pixel_height, w // aggregate_pixel_width

            # Initialize aggregated array
            aggregated_array = np.zeros((aggregate_pixel_height, aggregate_pixel_width, c), dtype=np.float32)

            # Perform pixel aggregation
            for m in range(aggregate_pixel_height):
                for n in range(aggregate_pixel_width):
                    y_start, y_end = m * block_h, (m + 1) * block_h
                    x_start, x_end = n * block_w, (n + 1) * block_w

                    # Average pooling
                    block = image_segment[y_start:y_end, x_start:x_end]
                    aggregated_array[m, n] = np.mean(block, axis=(0, 1))

            # print every 1000th apple
            if index % 1000 == 0:
                aggregated_array = aggregated_array.astype(np.float32) / 65535.0
                aggregated_array = (aggregated_array - aggregated_array.min()) / (aggregated_array.max() - aggregated_array.min())
                print(f'Max value: {np.max(aggregated_array)}, Min value: {np.min(aggregated_array)}')
                red_band = aggregated_array[:, :, 69]
                green_band = aggregated_array[:, :, 52]
                blue_band = aggregated_array[:, :, 18]

                rgb_image = np.stack([red_band, green_band, blue_band], axis=-1)

                # Display image
                plt.imshow(rgb_image)
                plt.axis("off")
                plt.title("RGB Composite from Hyperspectral Data extracted from tensor")
                plt.show()



            # Append results
            processed_images.append(aggregated_array)
            file_metadata.append(row)

             # Free memory after processing
            del image_segment  
            gc.collect() 

    return processed_images, file_metadata


# In[ ]:


# test = data_NZ2023_a_final2[1:2]
# print(test)

# process_spectral_images(test, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_NZ_2023', side='a', aggregate_pixel_height = 50, aggregate_pixel_width = 50)


# In[ ]:


# # Images at 20x20
# images_NZ2023_a, files_NZ2023_a = process_spectral_images(data_NZ2023_a_final2, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_NZ_2023', side='a', aggregate_pixel_height = 20, aggregate_pixel_width = 20)
# images_NZ2023_b, files_NZ2023_b = process_spectral_images(data_NZ2023_b_final2, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_NZ_2023', side='b', aggregate_pixel_height = 20, aggregate_pixel_width = 20)
# images_NZ2023_c, files_NZ2023_c = process_spectral_images(data_NZ2023_c_final2, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_NZ_2023', side='c', aggregate_pixel_height = 20, aggregate_pixel_width = 20)
# images_NZ2023_d, files_NZ2023_d = process_spectral_images(data_NZ2023_d_final2, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_NZ_2023', side='d', aggregate_pixel_height = 20, aggregate_pixel_width = 20)

# Images at 50x50
images_NZ2023_a, files_NZ2023_a = process_spectral_images(data_NZ2023_a_final2, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_NZ_2023', side='a', aggregate_pixel_height = 50, aggregate_pixel_width = 50)
images_NZ2023_b, files_NZ2023_b = process_spectral_images(data_NZ2023_b_final2, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_NZ_2023', side='b', aggregate_pixel_height = 50, aggregate_pixel_width = 50)
images_NZ2023_c, files_NZ2023_c = process_spectral_images(data_NZ2023_c_final2, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_NZ_2023', side='c', aggregate_pixel_height = 50, aggregate_pixel_width = 50)
images_NZ2023_d, files_NZ2023_d = process_spectral_images(data_NZ2023_d_final2, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_NZ_2023', side='d', aggregate_pixel_height = 50, aggregate_pixel_width = 50)






# In[17]:


counter = 1
def concatenate_with_counter(row):
    global counter
    result = f"{row['ID']}_{counter}"
    counter += 1
    return result


# In[18]:


files_NZ2023_a = pd.DataFrame(files_NZ2023_a)
files_NZ2023_a.explode('sorted_tensor').reset_index(drop=True)

files_NZ2023_b = pd.DataFrame(files_NZ2023_b)
files_NZ2023_b.explode('sorted_tensor').reset_index(drop=True)

files_NZ2023_c = pd.DataFrame(files_NZ2023_c)
files_NZ2023_c.explode('sorted_tensor').reset_index(drop=True)

files_NZ2023_d = pd.DataFrame(files_NZ2023_d)
files_NZ2023_d.explode('sorted_tensor').reset_index(drop=True)




# In[19]:


# save the metadata and processed images 

files_NZ2023_a['img_ID'] = files_NZ2023_a.apply(concatenate_with_counter, axis=1)
files_NZ2023_a.to_pickle(f'{pickle_path}aggregated_files_NZ2023_a.pkl')

files_NZ2023_b['img_ID'] = files_NZ2023_b.apply(concatenate_with_counter, axis=1)
files_NZ2023_b.to_pickle(f'{pickle_path}aggregated_files_NZ2023_b.pkl')

files_NZ2023_c['img_ID'] = files_NZ2023_c.apply(concatenate_with_counter, axis=1)
files_NZ2023_c.to_pickle(f'{pickle_path}aggregated_files_NZ2023_c.pkl')


files_NZ2023_d['img_ID'] = files_NZ2023_d.apply(concatenate_with_counter, axis=1)
files_NZ2023_d.to_pickle(f'{pickle_path}aggregated_files_NZ2023_d.pkl')




# In[20]:


print(files_NZ2023_a)


# In[ ]:


#save 20x20 images
spectral_folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Data_outputs_NZ2023/Spectral/subsetted_aggregated_hyperspectral_images/'

# Save 50x50 images
# spectral_folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Data_outputs_NZ2023/Spectral/aggregated_50px_HSI/'



# In[ ]:


#20x20px
for i, arr in enumerate(images_NZ2023_a):
    file_name = files_NZ2023_a.iloc[i]['img_ID']
    
    print(f"Saving {file_name}...")
    np.save(f"{spectral_folder_path}NZ2023/20px_{file_name}.npy", arr)

for i, arr in enumerate(images_NZ2023_b):
    file_name = files_NZ2023_b.iloc[i]['img_ID']
    
    print(f"Saving {file_name}...")
    np.save(f"{spectral_folder_path}NZ2023/20px_{file_name}.npy", arr)

for i, arr in enumerate(images_NZ2023_c):
    file_name = files_NZ2023_c.iloc[i]['img_ID']
    
    print(f"Saving {file_name}...")
    np.save(f"{spectral_folder_path}NZ2023/20px_{file_name}.npy", arr)

for i, arr in enumerate(images_NZ2023_d):
    file_name = files_NZ2023_d.iloc[i]['img_ID']
    
    print(f"Saving {file_name}...")
    np.save(f"{spectral_folder_path}NZ2023/20px_{file_name}.npy", arr)


# In[ ]:


# for i, arr in enumerate(images_NZ2023_a):
#     file_name = files_NZ2023_a.iloc[i]['img_ID']
    
#     print(f"Saving {file_name}...")
#     np.save(f"{spectral_folder_path}NZ2023/50px_{file_name}.npy", arr)

# for i, arr in enumerate(images_NZ2023_b):
#     file_name = files_NZ2023_b.iloc[i]['img_ID']
    
#     print(f"Saving {file_name}...")
#     np.save(f"{spectral_folder_path}NZ2023/50px_{file_name}.npy", arr)

# for i, arr in enumerate(images_NZ2023_c):
#     file_name = files_NZ2023_c.iloc[i]['img_ID']
    
#     print(f"Saving {file_name}...")
#     np.save(f"{spectral_folder_path}NZ2023/50px_{file_name}.npy", arr)

# for i, arr in enumerate(images_NZ2023_d):
#     file_name = files_NZ2023_d.iloc[i]['img_ID']
    
#     print(f"Saving {file_name}...")
#     np.save(f"{spectral_folder_path}NZ2023/50px_{file_name}.npy", arr)


# In[ ]:


# # repeat process for NZ2024 and UK2024

images_NZ2024_a, files_NZ2024_a = process_spectral_images(data_NZ2024_a_final2, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_NZ_2024', side='a', aggregate_pixel_height = 20, aggregate_pixel_width = 20)
images_NZ2024_b, files_NZ2024_b = process_spectral_images(data_NZ2024_b_final2, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_NZ_2024', side='b', aggregate_pixel_height = 20, aggregate_pixel_width = 20)
images_NZ2024_c, files_NZ2024_c = process_spectral_images(data_NZ2024_c_final2, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_NZ_2024', side='c', aggregate_pixel_height = 20, aggregate_pixel_width = 20)
images_NZ2024_d, files_NZ2024_d = process_spectral_images(data_NZ2024_d_final2, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_NZ_2024', side='d', aggregate_pixel_height = 20, aggregate_pixel_width = 20)

images_UK2024_a, files_UK2024_a = process_spectral_images(data_UK2024_a_final2, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_UK_2024', side='a', aggregate_pixel_height = 20, aggregate_pixel_width = 20)
images_UK2024_b, files_UK2024_b = process_spectral_images(data_UK2024_b_final2, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_UK_2024', side='b', aggregate_pixel_height = 20, aggregate_pixel_width = 20)
images_UK2024_c, files_UK2024_c = process_spectral_images(data_UK2024_c_final2, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_UK_2024', side='c', aggregate_pixel_height = 20, aggregate_pixel_width = 20)
images_UK2024_d, files_UK2024_d = process_spectral_images(data_UK2024_d_final2, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_UK_2024', side='d', aggregate_pixel_height = 20, aggregate_pixel_width = 20)



# repeat process for NZ2024 and UK2024

# images_NZ2024_a, files_NZ2024_a = process_spectral_images(data_NZ2024_a_final2, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_NZ_2024', side='a', aggregate_pixel_height = 50, aggregate_pixel_width = 50)
# images_NZ2024_b, files_NZ2024_b = process_spectral_images(data_NZ2024_b_final2, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_NZ_2024', side='b', aggregate_pixel_height = 50, aggregate_pixel_width = 50)
# images_NZ2024_c, files_NZ2024_c = process_spectral_images(data_NZ2024_c_final2, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_NZ_2024', side='c', aggregate_pixel_height = 50, aggregate_pixel_width = 50)
# images_NZ2024_d, files_NZ2024_d = process_spectral_images(data_NZ2024_d_final2, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_NZ_2024', side='d', aggregate_pixel_height = 50, aggregate_pixel_width = 50)

# images_UK2024_a, files_UK2024_a = process_spectral_images(data_UK2024_a_final2, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_UK_2024', side='a', aggregate_pixel_height = 50, aggregate_pixel_width = 50)
# images_UK2024_b, files_UK2024_b = process_spectral_images(data_UK2024_b_final2, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_UK_2024', side='b', aggregate_pixel_height = 50, aggregate_pixel_width = 50)
# images_UK2024_c, files_UK2024_c = process_spectral_images(data_UK2024_c_final2, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_UK_2024', side='c', aggregate_pixel_height = 50, aggregate_pixel_width = 50)
# images_UK2024_d, files_UK2024_d = process_spectral_images(data_UK2024_d_final2, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_UK_2024', side='d', aggregate_pixel_height = 50, aggregate_pixel_width = 50)





# In[24]:


files_NZ2024_a = pd.DataFrame(files_NZ2024_a)
files_NZ2024_a.explode('sorted_tensor').reset_index(drop=True)
files_NZ2024_b = pd.DataFrame(files_NZ2024_b)
files_NZ2024_b.explode('sorted_tensor').reset_index(drop=True)
files_NZ2024_c = pd.DataFrame(files_NZ2024_c)
files_NZ2024_c.explode('sorted_tensor').reset_index(drop=True)
files_NZ2024_d = pd.DataFrame(files_NZ2024_d)
files_NZ2024_d.explode('sorted_tensor').reset_index(drop=True)

files_UK2024_a = pd.DataFrame(files_UK2024_a)
files_UK2024_a.explode('sorted_tensor').reset_index(drop=True)
files_UK2024_b = pd.DataFrame(files_UK2024_b)
files_UK2024_b.explode('sorted_tensor').reset_index(drop=True)
files_UK2024_c = pd.DataFrame(files_UK2024_c)
files_UK2024_c.explode('sorted_tensor').reset_index(drop=True)
files_UK2024_d = pd.DataFrame(files_UK2024_d)
files_UK2024_d.explode('sorted_tensor').reset_index(drop=True)


# In[25]:


# save the metadata and processed images 
# Repeat for NZ2024 and UK2024

files_NZ2024_a['img_ID'] = files_NZ2024_a.apply(concatenate_with_counter, axis=1)
files_NZ2024_a.to_pickle(f'{pickle_path}aggregated_files_NZ2024_a.pkl')
files_NZ2024_b['img_ID'] = files_NZ2024_b.apply(concatenate_with_counter, axis=1)
files_NZ2024_b.to_pickle(f'{pickle_path}aggregated_files_NZ2024_b.pkl')
files_NZ2024_c['img_ID'] = files_NZ2024_c.apply(concatenate_with_counter, axis=1)
files_NZ2024_c.to_pickle(f'{pickle_path}aggregated_files_NZ2024_c.pkl')
files_NZ2024_d['img_ID'] = files_NZ2024_d.apply(concatenate_with_counter, axis=1)
files_NZ2024_d.to_pickle(f'{pickle_path}aggregated_files_NZ2024_d.pkl')


files_UK2024_a['img_ID'] = files_UK2024_a.apply(concatenate_with_counter, axis=1)
files_UK2024_a.to_pickle(f'{pickle_path}aggregated_files_UK2024_a.pkl')
files_UK2024_b['img_ID'] = files_UK2024_b.apply(concatenate_with_counter, axis=1)
files_UK2024_b.to_pickle(f'{pickle_path}aggregated_files_UK2024_b.pkl')
files_UK2024_c['img_ID'] = files_UK2024_c.apply(concatenate_with_counter, axis=1)
files_UK2024_c.to_pickle(f'{pickle_path}aggregated_files_UK2024_c.pkl')
files_UK2024_d['img_ID'] = files_UK2024_d.apply(concatenate_with_counter, axis=1)
files_UK2024_d.to_pickle(f'{pickle_path}aggregated_files_UK2024_d.pkl')


# In[ ]:


#20x20px
for i, arr in enumerate(images_NZ2024_a):
    file_name = files_NZ2024_a.iloc[i]['img_ID']
    print(f"Saving {file_name}...")
    np.save(f"{spectral_folder_path}NZ2024/20px_{file_name}.npy", arr)

for i, arr in enumerate(images_NZ2024_b):
    file_name = files_NZ2024_b.iloc[i]['img_ID']
    print(f"Saving {file_name}...")
    np.save(f"{spectral_folder_path}NZ2024/20px_{file_name}.npy", arr)

for i, arr in enumerate(images_NZ2024_c):
    file_name = files_NZ2024_c.iloc[i]['img_ID']
    print(f"Saving {file_name}...")
    np.save(f"{spectral_folder_path}NZ2024/20px_{file_name}.npy", arr)

for i, arr in enumerate(images_NZ2024_d):
    file_name = files_NZ2024_d.iloc[i]['img_ID']
    print(f"Saving {file_name}...")
    np.save(f"{spectral_folder_path}NZ2024/20px_{file_name}.npy", arr)


# # for 50x50px images
# for i, arr in enumerate(images_NZ2024_a):
#     file_name = files_NZ2024_a.iloc[i]['img_ID']
#     print(f"Saving {file_name}...")
#     np.save(f"{spectral_folder_path}NZ2024/50px_{file_name}.npy", arr)

# for i, arr in enumerate(images_NZ2024_b):
#     file_name = files_NZ2024_b.iloc[i]['img_ID']
#     print(f"Saving {file_name}...")
#     np.save(f"{spectral_folder_path}NZ2024/50px_{file_name}.npy", arr)

# for i, arr in enumerate(images_NZ2024_c):
#     file_name = files_NZ2024_c.iloc[i]['img_ID']
#     print(f"Saving {file_name}...")
#     np.save(f"{spectral_folder_path}NZ2024/50px_{file_name}.npy", arr)

# for i, arr in enumerate(images_NZ2024_d):
#     file_name = files_NZ2024_d.iloc[i]['img_ID']
#     print(f"Saving {file_name}...")
#     np.save(f"{spectral_folder_path}NZ2024/50px_{file_name}.npy", arr)


# In[ ]:


#20x20 px
for i, arr in enumerate(images_UK2024_a):
    file_name = files_UK2024_a.iloc[i]['img_ID']
    print(f"Saving {file_name}...")
    np.save(f"{spectral_folder_path}UK2024/20px_{file_name}.npy", arr)

for i, arr in enumerate(images_UK2024_b):
    file_name = files_UK2024_b.iloc[i]['img_ID']
    print(f"Saving {file_name}...")
    np.save(f"{spectral_folder_path}UK2024/20px_{file_name}.npy", arr)

for i, arr in enumerate(images_UK2024_c):
    file_name = files_UK2024_c.iloc[i]['img_ID']
    print(f"Saving {file_name}...")
    np.save(f"{spectral_folder_path}UK2024/20px_{file_name}.npy", arr)

for i, arr in enumerate(images_UK2024_d):
    file_name = files_UK2024_d.iloc[i]['img_ID']
    print(f"Saving {file_name}...")
    np.save(f"{spectral_folder_path}UK2024/20px_{file_name}.npy", arr)


# # # Save 50x50 images
# for i, arr in enumerate(images_UK2024_a):
#     file_name = files_UK2024_a.iloc[i]['img_ID']
#     print(f"Saving {file_name}...")
#     np.save(f"{spectral_folder_path}UK2024/50px_{file_name}.npy", arr)

# for i, arr in enumerate(images_UK2024_b):
#     file_name = files_UK2024_b.iloc[i]['img_ID']
#     print(f"Saving {file_name}...")
#     np.save(f"{spectral_folder_path}UK2024/50px_{file_name}.npy", arr)

# for i, arr in enumerate(images_UK2024_c):
#     file_name = files_UK2024_c.iloc[i]['img_ID']
#     print(f"Saving {file_name}...")
#     np.save(f"{spectral_folder_path}UK2024/50px_{file_name}.npy", arr)

# for i, arr in enumerate(images_UK2024_d):
#     file_name = files_UK2024_d.iloc[i]['img_ID']
#     print(f"Saving {file_name}...")
#     np.save(f"{spectral_folder_path}UK2024/50px_{file_name}.npy", arr)


# In[ ]:




