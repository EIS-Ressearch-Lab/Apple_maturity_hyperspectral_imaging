#!/usr/bin/env python
# coding: utf-8

# In[36]:


import torch
print(torch.cuda.is_available())


from detecto import core, utils, visualize
from detecto.visualize import show_labeled_image, plot_prediction_grid
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import pandas as pd
import glob
import re
import os
import gc
import cv2

from Function_definitions import process_spectral_images, xywh_to_xyxy

pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 10)


# In[37]:


gc.collect()


# In[ ]:


# # load data_final to pickle
pickle_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Pickle_files/Tensors/'

# data_NZ2023_a_final = pd.read_pickle(f'{pickle_path}data_NZ2023_a_final.pkl')
# data_NZ2023_b_final = pd.read_pickle(f'{pickle_path}data_NZ2023_b_final.pkl')
# data_NZ2023_c_final = pd.read_pickle(f'{pickle_path}data_NZ2023_c_final.pkl')
# data_NZ2023_d_final = pd.read_pickle(f'{pickle_path}data_NZ2023_d_final.pkl')
# data_NZ2024_a_final = pd.read_pickle(f'{pickle_path}data_NZ2024_a_final.pkl')
# data_NZ2024_b_final = pd.read_pickle(f'{pickle_path}data_NZ2024_b_final.pkl')
# data_NZ2024_c_final = pd.read_pickle(f'{pickle_path}data_NZ2024_c_final.pkl')
# data_NZ2024_d_final = pd.read_pickle(f'{pickle_path}data_NZ2024_d_final.pkl')
# data_UK2024_a_final = pd.read_pickle(f'{pickle_path}data_UK2024_a_final.pkl')
# data_UK2024_b_final = pd.read_pickle(f'{pickle_path}data_UK2024_b_final.pkl')
# data_UK2024_c_final = pd.read_pickle(f'{pickle_path}data_UK2024_c_final.pkl')
# data_UK2024_d_final = pd.read_pickle(f'{pickle_path}data_UK2024_d_final.pkl')


# pickle_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Pickle_files/Tensors/'

# Generate the missing images
# Load missing data
data_NZ2023_a_final3 = pd.read_pickle(f'{pickle_path}data_NZ2023_a_final3.pkl')
data_NZ2023_b_final3 = pd.read_pickle(f'{pickle_path}data_NZ2023_b_final3.pkl')
data_NZ2023_c_final3 = pd.read_pickle(f'{pickle_path}data_NZ2023_c_final3.pkl')
data_NZ2023_d_final3 = pd.read_pickle(f'{pickle_path}data_NZ2023_d_final3.pkl')
data_NZ2024_a_final3 = pd.read_pickle(f'{pickle_path}data_NZ2024_a_final3.pkl')
data_NZ2024_b_final3 = pd.read_pickle(f'{pickle_path}data_NZ2024_b_final3.pkl')
data_NZ2024_c_final3 = pd.read_pickle(f'{pickle_path}data_NZ2024_c_final3.pkl')
data_NZ2024_d_final3 = pd.read_pickle(f'{pickle_path}data_NZ2024_d_final3.pkl')
data_UK2024_a_final3 = pd.read_pickle(f'{pickle_path}data_UK2024_a_final3.pkl')
data_UK2024_b_final3 = pd.read_pickle(f'{pickle_path}data_UK2024_b_final3.pkl')
data_UK2024_c_final3 = pd.read_pickle(f'{pickle_path}data_UK2024_c_final3.pkl')
data_UK2024_d_final3 = pd.read_pickle(f'{pickle_path}data_UK2024_d_final3.pkl')




# In[ ]:


data_NZ2023_a_final = data_NZ2023_a_final3
data_NZ2023_b_final = data_NZ2023_b_final3
data_NZ2023_c_final = data_NZ2023_c_final3
data_NZ2023_d_final = data_NZ2023_d_final3
data_NZ2024_a_final = data_NZ2024_a_final3
data_NZ2024_b_final = data_NZ2024_b_final3
data_NZ2024_c_final = data_NZ2024_c_final3
data_NZ2024_d_final = data_NZ2024_d_final3
data_UK2024_a_final = data_UK2024_a_final3
data_UK2024_b_final = data_UK2024_b_final3
data_UK2024_c_final = data_UK2024_c_final3
data_UK2024_d_final = data_UK2024_d_final3


# In[ ]:


# print(data_NZ2023_a_final.head())
# print(data_NZ2024_b_final.head())
# print(data_UK2024_c_final.head())
# print(data_UK2024_d_final['Spectral_folder_D'][0])


# In[ ]:


# print(len(data_NZ2023_a_final))
# print(len(data_NZ2023_b_final))
# print(len(data_NZ2023_c_final))
# print(len(data_NZ2023_d_final))
# print(len(data_NZ2024_a_final))
# print(len(data_NZ2024_b_final))
# print(len(data_NZ2024_c_final))
# print(len(data_NZ2024_d_final))
# print(len(data_UK2024_a_final))
# print(len(data_UK2024_b_final))
# print(len(data_UK2024_c_final))
# print(len(data_UK2024_d_final))


# In[18]:


# Spectral folder

# Split by '/' and remove last split in string



# data_NZ2023_a_final['Spectral_folder_a'] = data_NZ2023_a_final['Image_folder_A'].str.split('/').str[:-1].str.join('/')
# data_NZ2023_b_final['Spectral_folder_b'] = data_NZ2023_b_final['Image_folder_B'].str.split('/').str[:-1].str.join('/')
# data_NZ2023_c_final['Spectral_folder_c'] = data_NZ2023_c_final['Image_folder_C'].str.split('/').str[:-1].str.join('/')
# data_NZ2023_d_final['Spectral_folder_d'] = data_NZ2023_d_final['Image_folder_D'].str.split('/').str[:-1].str.join('/')
# data_NZ2024_a_final['Spectral_folder_a'] = data_NZ2024_a_final['Image_folder_A'].str.split('/').str[:-1].str.join('/')
# data_NZ2024_b_final['Spectral_folder_b'] = data_NZ2024_b_final['Image_folder_B'].str.split('/').str[:-1].str.join('/')
# data_NZ2024_c_final['Spectral_folder_c'] = data_NZ2024_c_final['Image_folder_C'].str.split('/').str[:-1].str.join('/')
# data_NZ2024_d_final['Spectral_folder_d'] = data_NZ2024_d_final['Image_folder_D'].str.split('/').str[:-1].str.join('/')
# data_UK2024_a_final['Spectral_folder_a'] = data_UK2024_a_final['Image_folder_A'].str.split('/').str[:-1].str.join('/')
# data_UK2024_b_final['Spectral_folder_b'] = data_UK2024_b_final['Image_folder_B'].str.split('/').str[:-1].str.join('/')
# data_UK2024_c_final['Spectral_folder_c'] = data_UK2024_c_final['Image_folder_C'].str.split('/').str[:-1].str.join('/')
# data_UK2024_d_final['Spectral_folder_d'] = data_UK2024_d_final['Image_folder_D'].str.split('/').str[:-1].str.join('/')


# In[19]:


# print(data_NZ2023_a_final['Spectral_folder_a'][0])
# print(data_NZ2024_b_final['Spectral_folder_b'][0])
# print(data_UK2024_c_final['Spectral_folder_c'][0])


# In[ ]:


# test = data_UK2024_d_final[:4]
# print(test)
# print(test['Spectral_folder_D'][0])


# In[ ]:


# folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/'
# loc_link = 'spectral_UK_2024/'
# img_path = glob.glob(f'{folder_path}{loc_link}{test["Spectral_folder_D"][0]}/REFLECTANCE*.png')
# print(img_path[0])
# original_img = utils.read_image(img_path[0])
# plt.imshow(original_img)


# In[22]:


# # Code to check that the function works correctly

# side = 'D'
# height=512
# width=512
# aggregate_pixel_height=30, 
# aggregate_pixel_width=30
# df = test
# folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/'
# loc_link = 'spectral_UK_2024/'

# processed_images = []
# file_metadata = []

# fig, axes = plt.subplots(len(df),2, figsize=(10, len(df)*5))
                         
# for index, row in df.iterrows():
#     print(f"Processing {index + 1}/{len(df)}")

#     spectral_link = row[f'Spectral_folder_{side}']
#     dat_file = glob.glob(f"{folder_path}{loc_link}{spectral_link}/*.dat")

#     if not dat_file:
#             print(f"No .dat file found in {spectral_link}. Skipping...")
#             continue

#     # Load Spectral Data
#     spec_data = np.fromfile(dat_file[0], dtype=np.float32)
#     bands = len(spec_data) // (height * width)
#     spec_data = spec_data.reshape(height, bands, width)
#     spec_data = np.transpose(spec_data, (2, 0, 1))  # Convert to (H, W, B)
#     spectral_flipped = np.fliplr(spec_data)

#     spectral_flipped_normalised = (spectral_flipped - np.min(spectral_flipped)) / (np.max(spectral_flipped) - np.min(spectral_flipped)) * 255
#     spectral_flipped_normalised = spectral_flipped_normalised.astype(np.uint8)


#     mask = row['tensors']['segmentation']
#     img_with_mask = spectral_flipped.copy()
#     img_with_mask[mask == 0] = 0

#     img_with_mask_normalised = (img_with_mask - np.min(img_with_mask)) / (np.max(img_with_mask) - np.min(img_with_mask)) * 255
#     img_with_mask_normalised = img_with_mask_normalised.astype(np.uint8)


#     x, y, w, h = row['tensors']['bbox']
#     x1 = int(x)
#     y1 = int(y)
#     x2 = int(x + w)
#     y2 = int(y + h)

#     bbox = np.array([x1, y1, x2, y2])

#     #Compare original with segmented
#     red_band2 = spectral_flipped_normalised[:, :, 69]
#     green_band2 = spectral_flipped_normalised[:, :, 52]
#     blue_band2 = spectral_flipped_normalised[:, :, 18]
#     rgb_image2 = np.stack([red_band2, green_band2, blue_band2], axis=-1)

#     # Segemented images
#     red_band = img_with_mask_normalised[:, :, 69]
#     green_band = img_with_mask_normalised[:, :, 52]
#     blue_band = img_with_mask_normalised[:, :, 18]
#     rgb_image = np.stack([red_band, green_band, blue_band], axis=-1)

#     image_segment = rgb_image[y1:y2, x1:x2]
#     image_segment2 = rgb_image2[y1:y2, x1:x2]

    
#     # axes[index].imshow(image_segment)
#     # axes[index].set_title(f"Image {index + 1}")

    
#     # Plot the segmented image
#     axes[index, 0].imshow(image_segment)
#     axes[index, 0].set_title(f"Segmented Image {index + 1}")

#     # Plot the non-segmented image
#     axes[index, 1].imshow(image_segment2)
#     axes[index, 1].set_title(f"Non-Segmented Image {index + 1}")



#     # plt.imshow(image_segment)
#     # plt.show()

#     # axes.imshow(rgb_image)
# # Segment the image


# plt.tight_layout()
# plt.show



# In[40]:


# folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/'
# loc_link = 'spectral_UK_2024/'

# test_res, files = process_spectral_images(test, side='D', folder_path=folder_path, loc_link = 'spectral_UK_2024/', height=512, width=512, aggregate_pixel_height=30, aggregate_pixel_width=30)


# In[24]:


# print(test_res[1])


# In[32]:


img_size = 30
# img_size = 50


# In[ ]:


folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/'


# test_res = process_spectral_images(test, side='D', folder_path=folder_path, loc_link = 'spectral_UK_2024/', height=512, width=512, aggregate_pixel_height=30, aggregate_pixel_width=30)

if img_size == 30:
# Images at 30x30
    images_NZ2023_a, files_NZ2023_a = process_spectral_images(data_NZ2023_a_final, folder_path = folder_path, loc_link = 'spectral_NZ_2023/', side='A', aggregate_pixel_height = img_size, aggregate_pixel_width = img_size)
    images_NZ2023_b, files_NZ2023_b = process_spectral_images(data_NZ2023_b_final, folder_path = folder_path, loc_link = 'spectral_NZ_2023/', side='B', aggregate_pixel_height = img_size, aggregate_pixel_width = img_size)
    images_NZ2023_c, files_NZ2023_c = process_spectral_images(data_NZ2023_c_final, folder_path = folder_path, loc_link = 'spectral_NZ_2023/', side='C', aggregate_pixel_height = img_size, aggregate_pixel_width = img_size)
    images_NZ2023_d, files_NZ2023_d = process_spectral_images(data_NZ2023_d_final, folder_path = folder_path, loc_link = 'spectral_NZ_2023/', side='D', aggregate_pixel_height = img_size, aggregate_pixel_width = img_size)
elif img_size == 50:
    # Images at 50x50
    images_NZ2023_a, files_NZ2023_a = process_spectral_images(data_NZ2023_a_final, folder_path = folder_path, loc_link = 'spectral_NZ_2023/', side='A', aggregate_pixel_height = img_size, aggregate_pixel_width = img_size)
    images_NZ2023_b, files_NZ2023_b = process_spectral_images(data_NZ2023_b_final, folder_path = folder_path, loc_link = 'spectral_NZ_2023/', side='B', aggregate_pixel_height = img_size, aggregate_pixel_width = img_size)
    images_NZ2023_c, files_NZ2023_c = process_spectral_images(data_NZ2023_c_final, folder_path = folder_path, loc_link = 'spectral_NZ_2023/', side='C', aggregate_pixel_height = img_size, aggregate_pixel_width = img_size)
    images_NZ2023_d, files_NZ2023_d = process_spectral_images(data_NZ2023_d_final, folder_path = folder_path, loc_link = 'spectral_NZ_2023/', side='D', aggregate_pixel_height = img_size, aggregate_pixel_width = img_size)






# In[27]:


print(data_NZ2023_a_final[:1])


# In[ ]:


# counter = 1
counter = 21756 # Continue from the last number in the previous file
def concatenate_with_counter(row):
    global counter
    result = f"{row['ID']}_{counter}"
    counter += 1
    return result


# In[ ]:


# files_NZ2023_a = pd.DataFrame(files_NZ2023_a)
# files_NZ2023_a.explode('sorted_tensor').reset_index(drop=True)

# files_NZ2023_b = pd.DataFrame(files_NZ2023_b)
# files_NZ2023_b.explode('sorted_tensor').reset_index(drop=True)

# files_NZ2023_c = pd.DataFrame(files_NZ2023_c)
# files_NZ2023_c.explode('sorted_tensor').reset_index(drop=True)

# files_NZ2023_d = pd.DataFrame(files_NZ2023_d)
# files_NZ2023_d.explode('sorted_tensor').reset_index(drop=True)




# In[ ]:


# files = pd.read_pickle(f'/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Pickle_files/aggregated_files_NZ2023_a.pkl')

# print(files.head(4))


# In[ ]:


# # print(files[:3])

# files_pd = pd.DataFrame(files)
# print(files_pd[:3])


# In[ ]:


# files_pd['img_ID'] =files_pd.apply(concatenate_with_counter, axis=1)
# files_pd.to_pickle(f'{pickle_path}files_test.pkl')


# In[ ]:


# print(files_pd)


# In[ ]:


# save the metadata and processed images 

files_NZ2023_a = pd.DataFrame(files_NZ2023_a)
files_NZ2023_a['img_ID'] = files_NZ2023_a.apply(concatenate_with_counter, axis=1)
files_NZ2023_a.to_pickle(f'{pickle_path}aggregated_files_NZ2023_a.pkl')

files_NZ2023_b = pd.DataFrame(files_NZ2023_b)
files_NZ2023_b['img_ID'] = files_NZ2023_b.apply(concatenate_with_counter, axis=1)
files_NZ2023_b.to_pickle(f'{pickle_path}aggregated_files_NZ2023_b.pkl')

files_NZ2023_c = pd.DataFrame(files_NZ2023_c)
files_NZ2023_c['img_ID'] = files_NZ2023_c.apply(concatenate_with_counter, axis=1)
files_NZ2023_c.to_pickle(f'{pickle_path}aggregated_files_NZ2023_c.pkl')

files_NZ2023_d = pd.DataFrame(files_NZ2023_d)
files_NZ2023_d['img_ID'] = files_NZ2023_d.apply(concatenate_with_counter, axis=1)
files_NZ2023_d.to_pickle(f'{pickle_path}aggregated_files_NZ2023_d.pkl')




# In[53]:


if img_size ==30:
    #save 30x30 images
    spectral_folder_path = '/media/2tbdisk3/data/Haidee/May_2025_spectral_img/Spectral/30x/'
elif img_size ==50:    
    # Save 50x50 images
    spectral_folder_path = '/media/2tbdisk3/data/Haidee/May_2025_spectral_img/Spectral/50px/'



# In[ ]:


# for i, arr in enumerate(test_res):
#         file_name = files_pd.iloc[i]['img_ID']

#         print(f"Saving {file_name}...")
#         np.save(f"{spectral_folder_path}NZ2023/30px_{file_name}.npy", arr)


# In[ ]:


if img_size == 30:
    #30x30px
    for i, arr in enumerate(images_NZ2023_a):
        file_name = files_NZ2023_a.iloc[i]['img_ID']

        print(f"Saving {file_name}...")
        np.save(f"{spectral_folder_path}NZ2023/30px_{file_name}.npy", arr)

    for i, arr in enumerate(images_NZ2023_b):
        file_name = files_NZ2023_b.iloc[i]['img_ID']

        print(f"Saving {file_name}...")
        np.save(f"{spectral_folder_path}NZ2023/30px_{file_name}.npy", arr)

    for i, arr in enumerate(images_NZ2023_c):
        file_name = files_NZ2023_c.iloc[i]['img_ID']

        print(f"Saving {file_name}...")
        np.save(f"{spectral_folder_path}NZ2023/30px_{file_name}.npy", arr)

    for i, arr in enumerate(images_NZ2023_d):
        file_name = files_NZ2023_d.iloc[i]['img_ID']

        print(f"Saving {file_name}...")
        np.save(f"{spectral_folder_path}NZ2023/30px_{file_name}.npy", arr)

elif img_size == 50:
    for i, arr in enumerate(images_NZ2023_a):
        file_name = files_NZ2023_a.iloc[i]['img_ID']
    
        print(f"Saving {file_name}...")
        np.save(f"{spectral_folder_path}NZ2023/50px_{file_name}.npy", arr)

    for i, arr in enumerate(images_NZ2023_b):
        file_name = files_NZ2023_b.iloc[i]['img_ID']
        
        print(f"Saving {file_name}...")
        np.save(f"{spectral_folder_path}NZ2023/50px_{file_name}.npy", arr)
    
    for i, arr in enumerate(images_NZ2023_c):
        file_name = files_NZ2023_c.iloc[i]['img_ID']
        
        print(f"Saving {file_name}...")
        np.save(f"{spectral_folder_path}NZ2023/50px_{file_name}.npy", arr)
    
    for i, arr in enumerate(images_NZ2023_d):
        file_name = files_NZ2023_d.iloc[i]['img_ID']
        
        print(f"Saving {file_name}...")
        np.save(f"{spectral_folder_path}NZ2023/50px_{file_name}.npy", arr)


# In[ ]:


# # repeat process for NZ2024 and UK2024

if img_size == 30:
    images_NZ2024_a, files_NZ2024_a = process_spectral_images(data_NZ2024_a_final, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_NZ_2024', side='A', aggregate_pixel_height = img_size, aggregate_pixel_width = img_size)
    images_NZ2024_b, files_NZ2024_b = process_spectral_images(data_NZ2024_b_final, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_NZ_2024', side='B', aggregate_pixel_height = img_size, aggregate_pixel_width = img_size)
    images_NZ2024_c, files_NZ2024_c = process_spectral_images(data_NZ2024_c_final, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_NZ_2024', side='C', aggregate_pixel_height = img_size, aggregate_pixel_width = img_size)
    images_NZ2024_d, files_NZ2024_d = process_spectral_images(data_NZ2024_d_final, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_NZ_2024', side='D', aggregate_pixel_height = img_size, aggregate_pixel_width = img_size)

    images_UK2024_a, files_UK2024_a = process_spectral_images(data_UK2024_a_final, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_UK_2024', side='A', aggregate_pixel_height = img_size, aggregate_pixel_width = img_size)
    images_UK2024_b, files_UK2024_b = process_spectral_images(data_UK2024_b_final, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_UK_2024', side='B', aggregate_pixel_height = img_size, aggregate_pixel_width = img_size)
    images_UK2024_c, files_UK2024_c = process_spectral_images(data_UK2024_c_final, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_UK_2024', side='C', aggregate_pixel_height = img_size, aggregate_pixel_width = img_size)
    images_UK2024_d, files_UK2024_d = process_spectral_images(data_UK2024_d_final, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_UK_2024', side='D', aggregate_pixel_height = img_size, aggregate_pixel_width = img_size)


elif img_size == 50:
    # repeat process for NZ2024 and UK2024

    images_NZ2024_a, files_NZ2024_a = process_spectral_images(data_NZ2024_a_final, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_NZ_2024', side='A', aggregate_pixel_height = 50, aggregate_pixel_width = 50)
    images_NZ2024_b, files_NZ2024_b = process_spectral_images(data_NZ2024_b_final, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_NZ_2024', side='B', aggregate_pixel_height = 50, aggregate_pixel_width = 50)
    images_NZ2024_c, files_NZ2024_c = process_spectral_images(data_NZ2024_c_final, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_NZ_2024', side='C', aggregate_pixel_height = 50, aggregate_pixel_width = 50)
    images_NZ2024_d, files_NZ2024_d = process_spectral_images(data_NZ2024_d_final, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_NZ_2024', side='D', aggregate_pixel_height = 50, aggregate_pixel_width = 50)

    images_UK2024_a, files_UK2024_a = process_spectral_images(data_UK2024_a_final, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_UK_2024', side='A', aggregate_pixel_height = 50, aggregate_pixel_width = 50)
    images_UK2024_b, files_UK2024_b = process_spectral_images(data_UK2024_b_final, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_UK_2024', side='B', aggregate_pixel_height = 50, aggregate_pixel_width = 50)
    images_UK2024_c, files_UK2024_c = process_spectral_images(data_UK2024_c_final, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_UK_2024', side='C', aggregate_pixel_height = 50, aggregate_pixel_width = 50)
    images_UK2024_d, files_UK2024_d = process_spectral_images(data_UK2024_d_final, folder_path = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_UK_2024', side='D', aggregate_pixel_height = 50, aggregate_pixel_width = 50)





# In[ ]:


# files_NZ2024_a = pd.DataFrame(files_NZ2024_a)
# files_NZ2024_a.explode('sorted_tensor').reset_index(drop=True)
# files_NZ2024_b = pd.DataFrame(files_NZ2024_b)
# files_NZ2024_b.explode('sorted_tensor').reset_index(drop=True)
# files_NZ2024_c = pd.DataFrame(files_NZ2024_c)
# files_NZ2024_c.explode('sorted_tensor').reset_index(drop=True)
# files_NZ2024_d = pd.DataFrame(files_NZ2024_d)
# files_NZ2024_d.explode('sorted_tensor').reset_index(drop=True)

# files_UK2024_a = pd.DataFrame(files_UK2024_a)
# files_UK2024_a.explode('sorted_tensor').reset_index(drop=True)
# files_UK2024_b = pd.DataFrame(files_UK2024_b)
# files_UK2024_b.explode('sorted_tensor').reset_index(drop=True)
# files_UK2024_c = pd.DataFrame(files_UK2024_c)
# files_UK2024_c.explode('sorted_tensor').reset_index(drop=True)
# files_UK2024_d = pd.DataFrame(files_UK2024_d)
# files_UK2024_d.explode('sorted_tensor').reset_index(drop=True)


# In[ ]:


# save the metadata and processed images 
# Repeat for NZ2024 and UK2024

files_NZ2024_a = pd.DataFrame(files_NZ2024_a)
files_NZ2024_a['img_ID'] = files_NZ2024_a.apply(concatenate_with_counter, axis=1)
files_NZ2024_a.to_pickle(f'{pickle_path}aggregated_files_NZ2024_a.pkl')
files_NZ2024_b = pd.DataFrame(files_NZ2024_b)
files_NZ2024_b['img_ID'] = files_NZ2024_b.apply(concatenate_with_counter, axis=1)
files_NZ2024_b.to_pickle(f'{pickle_path}aggregated_files_NZ2024_b.pkl')
files_NZ2024_c = pd.DataFrame(files_NZ2024_c)
files_NZ2024_c['img_ID'] = files_NZ2024_c.apply(concatenate_with_counter, axis=1)
files_NZ2024_c.to_pickle(f'{pickle_path}aggregated_files_NZ2024_c.pkl')
files_NZ2024_d = pd.DataFrame(files_NZ2024_d)
files_NZ2024_d['img_ID'] = files_NZ2024_d.apply(concatenate_with_counter, axis=1)
files_NZ2024_d.to_pickle(f'{pickle_path}aggregated_files_NZ2024_d.pkl')

files_UK2024_a = pd.DataFrame(files_UK2024_a)
files_UK2024_a['img_ID'] = files_UK2024_a.apply(concatenate_with_counter, axis=1)
files_UK2024_a.to_pickle(f'{pickle_path}aggregated_files_UK2024_a.pkl')
files_UK2024_b = pd.DataFrame(files_UK2024_b)
files_UK2024_b['img_ID'] = files_UK2024_b.apply(concatenate_with_counter, axis=1)
files_UK2024_b.to_pickle(f'{pickle_path}aggregated_files_UK2024_b.pkl')
files_UK2024_c = pd.DataFrame(files_UK2024_c)
files_UK2024_c['img_ID'] = files_UK2024_c.apply(concatenate_with_counter, axis=1)
files_UK2024_c.to_pickle(f'{pickle_path}aggregated_files_UK2024_c.pkl')
files_UK2024_d = pd.DataFrame(files_UK2024_d)
files_UK2024_d['img_ID'] = files_UK2024_d.apply(concatenate_with_counter, axis=1)
files_UK2024_d.to_pickle(f'{pickle_path}aggregated_files_UK2024_d.pkl')


# In[ ]:


#30x30px
if img_size == 30:
    for i, arr in enumerate(images_NZ2024_a):
        file_name = files_NZ2024_a.iloc[i]['img_ID']
        print(f"Saving {file_name}...")
        np.save(f"{spectral_folder_path}NZ2024/30px_{file_name}.npy", arr)

    for i, arr in enumerate(images_NZ2024_b):
        file_name = files_NZ2024_b.iloc[i]['img_ID']
        print(f"Saving {file_name}...")
        np.save(f"{spectral_folder_path}NZ2024/30px_{file_name}.npy", arr)

    for i, arr in enumerate(images_NZ2024_c):
        file_name = files_NZ2024_c.iloc[i]['img_ID']
        print(f"Saving {file_name}...")
        np.save(f"{spectral_folder_path}NZ2024/30px_{file_name}.npy", arr)

    for i, arr in enumerate(images_NZ2024_d):
        file_name = files_NZ2024_d.iloc[i]['img_ID']
        print(f"Saving {file_name}...")
        np.save(f"{spectral_folder_path}NZ2024/30px_{file_name}.npy", arr)

elif img_size == 50:
    # for 50x50px images
    for i, arr in enumerate(images_NZ2024_a):
        file_name = files_NZ2024_a.iloc[i]['img_ID']
        print(f"Saving {file_name}...")
        np.save(f"{spectral_folder_path}NZ2024/50px_{file_name}.npy", arr)

    for i, arr in enumerate(images_NZ2024_b):
        file_name = files_NZ2024_b.iloc[i]['img_ID']
        print(f"Saving {file_name}...")
        np.save(f"{spectral_folder_path}NZ2024/50px_{file_name}.npy", arr)

    for i, arr in enumerate(images_NZ2024_c):
        file_name = files_NZ2024_c.iloc[i]['img_ID']
        print(f"Saving {file_name}...")
        np.save(f"{spectral_folder_path}NZ2024/50px_{file_name}.npy", arr)

    for i, arr in enumerate(images_NZ2024_d):
        file_name = files_NZ2024_d.iloc[i]['img_ID']
        print(f"Saving {file_name}...")
        np.save(f"{spectral_folder_path}NZ2024/50px_{file_name}.npy", arr)


# In[ ]:


if img_size == 30:
    #20x20 px
    for i, arr in enumerate(images_UK2024_a):
        file_name = files_UK2024_a.iloc[i]['img_ID']
        print(f"Saving {file_name}...")
        np.save(f"{spectral_folder_path}UK2024/30px_{file_name}.npy", arr)

    for i, arr in enumerate(images_UK2024_b):
        file_name = files_UK2024_b.iloc[i]['img_ID']
        print(f"Saving {file_name}...")
        np.save(f"{spectral_folder_path}UK2024/30px_{file_name}.npy", arr)

    for i, arr in enumerate(images_UK2024_c):
        file_name = files_UK2024_c.iloc[i]['img_ID']
        print(f"Saving {file_name}...")
        np.save(f"{spectral_folder_path}UK2024/30px_{file_name}.npy", arr)

    for i, arr in enumerate(images_UK2024_d):
        file_name = files_UK2024_d.iloc[i]['img_ID']
        print(f"Saving {file_name}...")
        np.save(f"{spectral_folder_path}UK2024/30px_{file_name}.npy", arr)

elif img_size == 50:
    # # Save 50x50 images
    for i, arr in enumerate(images_UK2024_a):
        file_name = files_UK2024_a.iloc[i]['img_ID']
        print(f"Saving {file_name}...")
        np.save(f"{spectral_folder_path}UK2024/50px_{file_name}.npy", arr)

    for i, arr in enumerate(images_UK2024_b):
        file_name = files_UK2024_b.iloc[i]['img_ID']
        print(f"Saving {file_name}...")
        np.save(f"{spectral_folder_path}UK2024/50px_{file_name}.npy", arr)

    for i, arr in enumerate(images_UK2024_c):
        file_name = files_UK2024_c.iloc[i]['img_ID']
        print(f"Saving {file_name}...")
        np.save(f"{spectral_folder_path}UK2024/50px_{file_name}.npy", arr)

    for i, arr in enumerate(images_UK2024_d):
        file_name = files_UK2024_d.iloc[i]['img_ID']
        print(f"Saving {file_name}...")
        np.save(f"{spectral_folder_path}UK2024/50px_{file_name}.npy", arr)


# In[ ]:


# # files_UK2024_d.iloc[66]["sorted_tensor"][3]
# print(files_UK2024_d.iloc[66]['Spectral_folder_D'])


# In[ ]:


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import glob
# from Function_definitions import xywh_to_xyxy
# # Checking the saved images are correct

# files_NZ2023_d = pd.read_pickle(f'/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Pickle_files/Tensors/aggregated_files_NZ2023_d.pkl')

# file_name = files_NZ2023_d.iloc[66]['img_ID']

# # original

# Folder_path_UK2024 = '/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/'
# loc_link = 'spectral_NZ_2023/'
# side = 'D'
# spectral_link = Folder_path_UK2024 + loc_link + files_NZ2023_d.iloc[66]['Spectral_folder_D']
# print(spectral_link)
# dat_files = glob.glob(f"{spectral_link}/*.dat")
# print(dat_files)

# height_original = 512
# width_original = 512

# # Load spectral data

# spec_data = np.fromfile(dat_files[0], dtype=np.float32)
# bands = len(spec_data) // (height_original * width_original)
# spec_data = spec_data.reshape(height_original, bands, width_original)
# spec_data = np.transpose(spec_data, (2, 0, 1))  # Convert to (H, W, B)
# spectral_flipped = np.fliplr(spec_data)


# x, y, h, w = files_NZ2023_d.iloc[66]["tensors"]['bbox']
# x1, y1, x2, y2 = xywh_to_xyxy(files_NZ2023_d.iloc[66]["tensors"]['bbox'])
# x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
# image_segment = spectral_flipped[y1:y2, x1:x2]
# # image_segment = image_segment / 65535.0
# image_segment = (image_segment - image_segment.min()) / (image_segment.max() - image_segment.min())
# print(f'Max value: {np.max(image_segment)}, Min value: {np.min(image_segment)}')
# red_band = image_segment[:, :, 70]
# green_band = image_segment[:, :, 53]
# blue_band = image_segment[:, :, 19]
# rgb_image = np.stack([red_band, green_band, blue_band], axis=-1)
# # Display image
# plt.imshow(rgb_image)
# plt.axis("off")
# plt.title("RGB Composite from Hyperspectral Data extracted from tensor - original")
# plt.show()




# # 30x30px images
# spectral_folder_path = '/media/2tbdisk3/data/Haidee/May_2025_spectral_img/Spectral/30x/'



# test_img = np.load(f"{spectral_folder_path}NZ2023/30px_{file_name}.npy")

# print(test_img.shape)

# print(test_img.max())


# red_band = test_img[:, :, 69]
# green_band = test_img[:, :, 52]
# blue_band = test_img[:, :, 18]
# rgb_image = np.stack([red_band, green_band, blue_band], axis=-1)
# # Display image
# plt.imshow(rgb_image)
# plt.axis("off")
# plt.title("RGB Composite from Hyperspectral Data extracted from tensor - 30x30xpx")
# plt.show()



# # 50x50 images
# spectral_folder_path = '/media/2tbdisk3/data/Haidee/May_2025_spectral_img/Spectral/50px/'

# test_img = np.load(f"{spectral_folder_path}NZ2023/50px_{file_name}.npy")


# print(test_img.shape)

# print(test_img.max())


# red_band = test_img[:, :, 69]
# green_band = test_img[:, :, 52]
# blue_band = test_img[:, :, 18]
# rgb_image = np.stack([red_band, green_band, blue_band], axis=-1)
# # Display image
# plt.imshow(rgb_image)
# plt.axis("off")
# plt.title("RGB Composite from Hyperspectral Data extracted from tensor - 50x50xpx")
# plt.show()




