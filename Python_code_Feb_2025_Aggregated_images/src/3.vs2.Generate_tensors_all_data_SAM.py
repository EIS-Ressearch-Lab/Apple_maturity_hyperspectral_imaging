#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
print(torch.cuda.is_available())


from detecto import core, utils, visualize
from detecto.visualize import show_labeled_image #, plot_prediction_grid
# from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from PIL import Image
import cv2
import random
import pickle


# In[2]:


# pd.set_option("display.max_colwidth", None)


# In[3]:


file_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/csv_files/All_data_csv/'

# load csv files
data_NZ2023_a_na = pd.read_csv(f'{file_path}data_NZ2023_a_na.csv')
data_NZ2023_b_na = pd.read_csv(f'{file_path}data_NZ2023_b_na.csv')
data_NZ2023_c_na = pd.read_csv(f'{file_path}data_NZ2023_c_na.csv')
data_NZ2023_d_na = pd.read_csv(f'{file_path}data_NZ2023_d_na.csv')
data_NZ2024_a_na = pd.read_csv(f'{file_path}data_NZ2024_a_na.csv')
data_NZ2024_b_na = pd.read_csv(f'{file_path}data_NZ2024_b_na.csv')
data_NZ2024_c_na = pd.read_csv(f'{file_path}data_NZ2024_c_na.csv')
data_NZ2024_d_na = pd.read_csv(f'{file_path}data_NZ2024_d_na.csv')
data_UK2024_a_na = pd.read_csv(f'{file_path}data_UK2024_a_na.csv')
data_UK2024_b_na = pd.read_csv(f'{file_path}data_UK2024_b_na.csv')
data_UK2024_c_na = pd.read_csv(f'{file_path}data_UK2024_c_na.csv')
data_UK2024_d_na = pd.read_csv(f'{file_path}data_UK2024_d_na.csv')



# In[4]:


# Spectral folder

# Split by '/' and remove last split in string

data_NZ2023_a_na['Spectral_folder_A'] = data_NZ2023_a_na['Image_folder_A'].str.split('/').str[:-1].str.join('/') + '/results/'
data_NZ2023_b_na['Spectral_folder_B'] = data_NZ2023_b_na['Image_folder_B'].str.split('/').str[:-1].str.join('/') + '/results/'
data_NZ2023_c_na['Spectral_folder_C'] = data_NZ2023_c_na['Image_folder_C'].str.split('/').str[:-1].str.join('/') + '/results/'
data_NZ2023_d_na['Spectral_folder_D'] = data_NZ2023_d_na['Image_folder_D'].str.split('/').str[:-1].str.join('/') + '/results/'
data_NZ2024_a_na['Spectral_folder_A'] = data_NZ2024_a_na['Image_folder_A'].str.split('/').str[:-1].str.join('/') + '/results/'
data_NZ2024_b_na['Spectral_folder_B'] = data_NZ2024_b_na['Image_folder_B'].str.split('/').str[:-1].str.join('/') + '/results/'
data_NZ2024_c_na['Spectral_folder_C'] = data_NZ2024_c_na['Image_folder_C'].str.split('/').str[:-1].str.join('/') + '/results/'
data_NZ2024_d_na['Spectral_folder_D'] = data_NZ2024_d_na['Image_folder_D'].str.split('/').str[:-1].str.join('/') + '/results/'
data_UK2024_a_na['Spectral_folder_A'] = data_UK2024_a_na['Image_folder_A'].str.split('/').str[:-1].str.join('/') + '/results/'
data_UK2024_b_na['Spectral_folder_B'] = data_UK2024_b_na['Image_folder_B'].str.split('/').str[:-1].str.join('/') + '/results/'
data_UK2024_c_na['Spectral_folder_C'] = data_UK2024_c_na['Image_folder_C'].str.split('/').str[:-1].str.join('/') + '/results/'
data_UK2024_d_na['Spectral_folder_D'] = data_UK2024_d_na['Image_folder_D'].str.split('/').str[:-1].str.join('/') + '/results/'


# In[5]:


# print(data_NZ2023_a_na[:7])


# In[2]:


import gc
gc.collect()

using_colab = False


# In[3]:


# SAM setup
# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )




# In[4]:


np.random.seed(3)

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)


# In[5]:


import sam2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

sam2_checkpoint = "/media/2tbdisk2/data/Haidee_apple_data/Haidee/SAM_checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

seed = 118
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

mask_generator_2 = SAM2AutomaticMaskGenerator(

    model=sam2,
    points_per_side=32,
    points_per_batch=50,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.95,
    stability_score_offset=0.89,
    crop_n_layers=1,
    box_nms_thresh=0, # low values for non-overlapping bboxes
    crop_n_points_downscale_factor=4,
    min_mask_region_area=9000.0,
    use_m2m=True,
)



# In[6]:


# Filter functions

def filter_circular_masks(masks, circularity_threshold=0.5):
    filtered_masks = []

    for mask_data in masks:
        mask = mask_data['segmentation'].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        area = cv2.contourArea(contour)
        circularity = 4 * np.pi * area / (perimeter ** 2)

        if circularity > circularity_threshold:
            filtered_masks.append(mask_data)

    return filtered_masks

def xywh_to_xyxy(bbox):
        x, y, w, h = bbox
        return np.array([x, y, x + w, y + h])


def remove_lower_overlapping_boxes(masks):
    
    def boxes_overlap(box1, box2):
        return not (box1[2] <= box2[0] or box1[0] >= box2[2] or
                    box1[3] <= box2[1] or box1[1] >= box2[3])

    to_remove = set()

    for i in range(len(masks)):
        if i in to_remove:
            continue
        box1 = xywh_to_xyxy(masks[i]['bbox'])

        for j in range(i + 1, len(masks)):
            if j in to_remove:
                continue
            box2 = xywh_to_xyxy(masks[j]['bbox'])

            if boxes_overlap(box1, box2):
                lower_idx = i if box1[3] > box2[3] else j
                to_remove.add(lower_idx)

    return [masks[i] for i in range(len(masks)) if i not in to_remove]



def plot_individual_masks(image, masks, bbox = False):
    """
    Plots each mask independently on the original image.

    Parameters:
    - image: NumPy array of the original image.
    - masks: List of dictionaries containing mask data.
    """
    num_objects = len(masks)
    fig, axes = plt.subplots(1, num_objects, figsize=(15, 5))

    for i, mask_data in enumerate(masks):
        mask = mask_data['segmentation']
        
        # Create a copy of the original image
        img_with_mask = image.copy()
        
        # Apply the mask to the image
        img_with_mask[mask == 0] = 0
        
        # Plot the image with the mask
        axes[i].imshow(img_with_mask)
        axes[i].set_title(f'Object {i+1}')
        axes[i].axis('off')

        if bbox == True:
            # Draw the bounding box
            x, y, w, h = mask_data['bbox']
            cv2.rectangle(img_with_mask, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)

            # Overlay the bounding box on the image
            axes[i].imshow(img_with_mask)

    plt.tight_layout()
    plt.show()

def reorganize_masks(masks, row_tolerance=30):
    """
    Reorganize SAM masks in a top-to-bottom, then left-to-right order.

    Parameters:
    - masks: List of dictionaries with 'bbox' keys.
    - row_tolerance: Max y-distance to group items into same row (in pixels).

    Returns:
    - Sorted list of masks.
    """

    # Extract bboxes and compute center points
    bboxes = [mask['bbox'] for mask in masks]
    # Convert to [x_min, y_min, x_max, y_max]
    bboxes_np = np.array([[x, y, x + w, y + h] for x, y, w, h in bboxes])
    y_centers = (bboxes_np[:, 1] + bboxes_np[:, 3]) / 2
    x_centers = (bboxes_np[:, 0] + bboxes_np[:, 2]) / 2

    # Combine data for sorting
    data = list(zip(bboxes_np, y_centers, x_centers, masks))

    # Step 1: Sort all by y_center
    data.sort(key=lambda x: x[1])  # sort top to bottom

    # Step 2: Group into rows using y_center proximity
    rows = []
    current_row = [data[0]]
    for i in range(1, len(data)):
        if abs(data[i][1] - current_row[-1][1]) < row_tolerance:
            current_row.append(data[i])
        else:
            rows.append(current_row)
            current_row = [data[i]]
    rows.append(current_row)

    # Step 3: Sort each row left to right (by x_center)
    for row in rows:
        row.sort(key=lambda x: x[2])  # sort left to right

    # Step 4: Flatten the result
    sorted_masks = [item[3] for row in rows for item in row]
    return sorted_masks


def process_stalks(sorted_masks, top_fraction=0.2, stalk_threshold=0.1):
    new_masks = []
    
    for mask_data in sorted_masks:
        mask = mask_data['segmentation']
        bbox = mask_data['bbox']
        
        x_min, y_min, width, height = map(int, bbox)
        x_max = x_min + width
        y_max = y_min + height

        # Work only inside the bbox area
        mask_crop = mask[y_min:y_max, x_min:x_max]
        h = mask_crop.shape[0]
        top_rows = int(h * top_fraction)

        top_region = mask_crop[:top_rows, :]
        body_region = mask_crop[top_rows:, :]

        top_pixels = np.sum(top_region)
        body_pixels = np.sum(body_region)

        # If top is small relative to body, treat as stalk
        if top_pixels > 0 and (top_pixels / (top_pixels + body_pixels)) < stalk_threshold:
            # print("Stalk detected, removing top region")
            mask_crop[:top_rows, :] = 0  # remove top part (stalk)

            new_mask = np.zeros_like(mask)
            new_mask[y_min:y_max, x_min:x_max] = mask_crop

            # Recalculate bbox
            y_coords, x_coords = np.where(new_mask)
            if len(y_coords) > 0:
                new_x_min, new_x_max = x_coords.min(), x_coords.max()
                new_y_min, new_y_max = y_coords.min(), y_coords.max()
                new_bbox = [
                    int(new_x_min),
                    int(new_y_min),
                    int(new_x_max - new_x_min),
                    int(new_y_max - new_y_min),
                ]
                new_masks.append({
                    'segmentation': new_mask,
                    'bbox': new_bbox,
                    'area': int(np.sum(new_mask)),
                })
        else:
            # Keep original if no stalk
            new_masks.append(mask_data)

    return new_masks







# In[4]:


# Using SAM segregation and filters to identify correct masks
def extractor(folder_link, loc_link, row, side, intensity_threshold=40, circularity_threshold=0.6):
    """
    Extract sorted, filtered SAM masks from an image in a specified folder.

    Args:
        folder_link (str): Base folder path.
        loc_link (str): Subdirectory path.
        folder_name (str): Folder name where image is stored.
        intensity_threshold (int): Minimum intensity to consider foreground.
        circularity_threshold (float): Circularity threshold for mask filtering.

    Returns:
        list: Sorted and filtered list of masks.
    """
     
    side = side 

    # Count = row['Count']
    folder_name = row[f'Spectral_folder_{side}']

    folder_path = f'{folder_link}{loc_link}{folder_name}'
    png_file = glob.glob(f'{folder_path}/REFLECTANCE*.png')
    if not png_file:
        return []
    
    rgb_image = utils.read_image(png_file[0])
    gen_mask = mask_generator_2.generate(rgb_image)
    # print(f"gen_mask count:{len(gen_mask)}")

    filtered_masks = []

    for mask_data in gen_mask:
        mask = mask_data['segmentation']
        bbox = mask_data['bbox']
        area = mask_data['area']
        width, height = bbox[2], bbox[3]
            # Intensity check
        if rgb_image[mask.astype(bool)].max() <= intensity_threshold:
            continue
        
        # Shape check
        if width > 1.5 * height:
            continue
        
        # Area check
        if area <= 5000:
            continue
        
        # Circularity check
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (contour_area / (perimeter ** 2))
        if circularity < circularity_threshold:
            continue
        
        filtered_masks.append(mask_data)
    if not filtered_masks:
        return []
    filtered_masks_no_base = remove_lower_overlapping_boxes(filtered_masks)
    filtered_masks_stalkless = process_stalks(filtered_masks_no_base, top_fraction=0.2)

    sorted_masks = reorganize_masks(filtered_masks_stalkless)
    

    return sorted_masks


# In[7]:


def lengther(data):
    return len(data)


# In[94]:


# Function to use extract function and return tensors in the correct format
def extractor_spectral(df_unique, folder_link, loc_link, side):
    """
    Extract tensors into each row of the dataframe
    Args:
        df_unique (DataFrame): DataFrame with unique rows.
        folder_link (str): Base folder path.
        loc_link (str): Subdirectory path.
        side (str): Side of the image (A, B, C, D).
    Returns:
        a pd.Series: Series of tensors to be combined into a DataFrame
    """
    
    # Apply the extractor function to each row
    df_unique['sorted_tensor'] = df_unique.apply(
        lambda row: extractor(folder_link, loc_link, row, side=side), axis=1)
    
    
    tensor_list = []
    total_length = len(df_unique)
    row_num = 1

    # Iterate for each row in the DataFrame and update tensor list
    for index, row in df_unique.iterrows():
        
        count = row['Count']
        sorted_tensors = row['sorted_tensor']

        if len(sorted_tensors) == count:
            print(f"Row {index}: Count matches length of sorted_tensor ({count})")
            tensor_list.append(sorted_tensors)
        else:
            print(f"Row {index}: Count does not match length of sorted_tensor (Count: {count}, Length: {len(sorted_tensors)})")
            tensor_list.append([{} for _ in range(count)])  # Append an empty dictionary if they don't match by the number of counts

        print(f"Completed {row_num}/{total_length} rows")
        row_num += 1

    flattened_list = [item for sublist in tensor_list for item in sublist]
    # tensor_list_pd = pd.Series(flattened_list)
    return flattened_list

      
    


# In[128]:


# Test dataset
# identify missing images due to poor mask generation

# CHeck if dictionary is empty
def is_empty_dict(d):
    return isinstance(d, dict) and not d





# In[129]:


def save_if_not_empty(list_name, list_data):
    if not list_data.empty:
        with open(f'{pickle_path}{list_name}.pkl', 'wb') as f:
            pickle.dump(list_data, f)


# In[16]:


# test_data = data_UK2024_a_na[:5]
# # 
# print(test_data)


# In[17]:


# # Test dataset # Generate counts of unique rows
# name_counts_test = test_data['Image_folder_A'].value_counts()
# # Create a new column with the counts
# test_data['Count'] = test_data['Image_folder_A'].map(name_counts_test)


# # Remove duplicates of Spectral_folder_A
# test_data_unique = test_data.drop_duplicates(subset=['Spectral_folder_A'])

# print(test_data_unique)


# In[18]:


# Generate counts of unique rows

# Side A files
name_counts_NZ2023 = data_NZ2023_a_na['Image_folder_A'].value_counts()
data_NZ2023_a_na['Count'] = data_NZ2023_a_na['Image_folder_A'].map(name_counts_NZ2023)
data_NZ2023_a_na_unique = data_NZ2023_a_na.drop_duplicates(subset=['Spectral_folder_A'])

name_counts_NZ2024 = data_NZ2024_a_na['Image_folder_A'].value_counts()
data_NZ2024_a_na['Count'] = data_NZ2024_a_na['Image_folder_A'].map(name_counts_NZ2024)
data_NZ2024_a_na_unique = data_NZ2024_a_na.drop_duplicates(subset=['Spectral_folder_A'])

name_counts_UK2024 = data_UK2024_a_na['Image_folder_A'].value_counts()
data_UK2024_a_na['Count'] = data_UK2024_a_na['Image_folder_A'].map(name_counts_UK2024)
data_UK2024_a_na_unique = data_UK2024_a_na.drop_duplicates(subset=['Spectral_folder_A'])

# Side B files
name_counts_NZ2023 = data_NZ2023_b_na['Image_folder_B'].value_counts()
data_NZ2023_b_na['Count'] = data_NZ2023_b_na['Image_folder_B'].map(name_counts_NZ2023)
data_NZ2023_b_na_unique = data_NZ2023_b_na.drop_duplicates(subset=['Spectral_folder_B'])

name_counts_NZ2024 = data_NZ2024_b_na['Image_folder_B'].value_counts()
data_NZ2024_b_na['Count'] = data_NZ2024_b_na['Image_folder_B'].map(name_counts_NZ2024)
data_NZ2024_b_na_unique = data_NZ2024_b_na.drop_duplicates(subset=['Spectral_folder_B'])

name_counts_UK2024 = data_UK2024_b_na['Image_folder_B'].value_counts()
data_UK2024_b_na['Count'] = data_UK2024_b_na['Image_folder_B'].map(name_counts_UK2024)
data_UK2024_b_na_unique = data_UK2024_b_na.drop_duplicates(subset=['Spectral_folder_B'])

# Side C files
name_counts_NZ2023 = data_NZ2023_c_na['Image_folder_C'].value_counts()
data_NZ2023_c_na['Count'] = data_NZ2023_c_na['Image_folder_C'].map(name_counts_NZ2023)
data_NZ2023_c_na_unique = data_NZ2023_c_na.drop_duplicates(subset=['Spectral_folder_C'])

name_counts_NZ2024 = data_NZ2024_c_na['Image_folder_C'].value_counts()
data_NZ2024_c_na['Count'] = data_NZ2024_c_na['Image_folder_C'].map(name_counts_NZ2024)
data_NZ2024_c_na_unique = data_NZ2024_c_na.drop_duplicates(subset=['Spectral_folder_C'])

name_counts_UK2024 = data_UK2024_c_na['Image_folder_C'].value_counts()
data_UK2024_c_na['Count'] = data_UK2024_c_na['Image_folder_C'].map(name_counts_UK2024)
data_UK2024_c_na_unique = data_UK2024_c_na.drop_duplicates(subset=['Spectral_folder_C'])

# Side D files
name_counts_NZ2023 = data_NZ2023_d_na['Image_folder_D'].value_counts()
data_NZ2023_d_na['Count'] = data_NZ2023_d_na['Image_folder_D'].map(name_counts_NZ2023)
data_NZ2023_d_na_unique = data_NZ2023_d_na.drop_duplicates(subset=['Spectral_folder_D'])

name_counts_NZ2024 = data_NZ2024_d_na['Image_folder_D'].value_counts()
data_NZ2024_d_na['Count'] = data_NZ2024_d_na['Image_folder_D'].map(name_counts_NZ2024)
data_NZ2024_d_na_unique = data_NZ2024_d_na.drop_duplicates(subset=['Spectral_folder_D'])

name_counts_UK2024 = data_UK2024_d_na['Image_folder_D'].value_counts()
data_UK2024_d_na['Count'] = data_UK2024_d_na['Image_folder_D'].map(name_counts_UK2024)
data_UK2024_d_na_unique = data_UK2024_d_na.drop_duplicates(subset=['Spectral_folder_D'])


# In[ ]:


# # Test function
# folder_link = "/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/"

# NZ2023_loc_link = 'spectral_NZ_2023/'
# NZ2024_loc_link = 'spectral_NZ_2024/'
# UK2024_loc_link = 'spectral_UK_2024/'

# test_data['tensors'] = extractor_spectral(test_data_unique, folder_link, UK2024_loc_link, side = 'A')
# # test = extractor_spectral(test_data_unique, folder_link, UK2024_loc_link, side = 'A')

# print(type(test_data))


# In[ ]:


# print(test_data)


# In[ ]:


# ## Check if the function worked
# folder_path = f'{folder_link}{UK2024_loc_link}{test_data["Spectral_folder_A"][0]}'
# png_file = glob.glob(f'{folder_path}/REFLECTANCE*.png')
# png_file = utils.read_image(png_file[0])

# for i, tensor in enumerate(test_data['tensors']):
#     if isinstance(tensor, dict) and tensor: # check if tensor is a non-empty dictionary
#         mask = tensor['segmentation']
#         img_with_mask = png_file.copy()
#         img_with_mask[mask == 0] = 0
#         x,y,w,h = tensor['bbox']
#         cv2.rectangle(img_with_mask, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
#         plt.imshow(img_with_mask)
#         plt.axis('off')
#         plt.title(f"Image {test_data['ID'][i]}")
#         plt.show()



# In[ ]:





# In[ ]:


#1118min30s
folder_link = "/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/"

NZ2023_loc_link = 'spectral_NZ_2023/'
NZ2024_loc_link = 'spectral_NZ_2024/'
UK2024_loc_link = 'spectral_UK_2024/'

# print(test_data)

# test_data['tensors'] = extractor_spectral(test_data_unique, folder_link, UK2024_loc_link, side = 'A')

data_NZ2023_a_na['tensors'] = extractor_spectral(data_NZ2023_a_na_unique, folder_link, NZ2023_loc_link, 'A')
data_NZ2023_b_na['tensors'] = extractor_spectral(data_NZ2023_b_na_unique, folder_link, NZ2023_loc_link, 'B')
data_NZ2023_c_na['tensors'] = extractor_spectral(data_NZ2023_c_na_unique, folder_link, NZ2023_loc_link, 'C') 
data_NZ2023_d_na['tensors'] = extractor_spectral(data_NZ2023_d_na_unique, folder_link, NZ2023_loc_link, 'D')
# data_NZ2023_a_na['sorted_tensor'].head(5)

data_NZ2024_a_na['tensors'] = extractor_spectral(data_NZ2024_a_na_unique, folder_link, NZ2024_loc_link, 'A')
data_NZ2024_b_na['tensors'] = extractor_spectral(data_NZ2024_b_na_unique, folder_link, NZ2024_loc_link, 'B')
data_NZ2024_c_na['tensors'] = extractor_spectral(data_NZ2024_c_na_unique, folder_link, NZ2024_loc_link, 'C') 
data_NZ2024_d_na['tensors'] = extractor_spectral(data_NZ2024_d_na_unique, folder_link, NZ2024_loc_link, 'D')
# data_NZ2024_a_na['sorted_tensor'].head(5)



data_UK2024_a_na['tensors'] = extractor_spectral(data_UK2024_a_na_unique, folder_link, UK2024_loc_link, 'A')
data_UK2024_b_na['tensors'] = extractor_spectral(data_UK2024_b_na_unique, folder_link, UK2024_loc_link, 'B')
data_UK2024_c_na['tensors'] = extractor_spectral(data_UK2024_c_na_unique, folder_link, UK2024_loc_link, 'C') 
data_UK2024_d_na['tensors'] = extractor_spectral(data_UK2024_d_na_unique, folder_link, UK2024_loc_link, 'D')
# data_UK2024_a_na['tensors'].head(5)


# In[ ]:


# rgb_image = utils.read_image('/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_NZ_2023/20230328 Fuji PFR orchard early pick/2023-03-28_002/2023-03-28_002.png')
# show_labeled_image(rgb_image, data_NZ2023_a_na['sorted_tensor'].iloc[0])
# # print(test_data['test_tensor'].iloc[0])

# dat_files = "/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/spectral_NZ_2023/20230328 Fuji PFR orchard early pick/2023-03-28_002/results/REFLECTANCE_2023-03-28_002.dat"

# height, width, bands = 512, 512, 204
# spectral_data = np.fromfile(dat_files, dtype=np.float32)
# bands = len(spectral_data) // (height * width)
# spectral_data = spectral_data.reshape(height, bands, width)
# spectral_data = np.transpose(spectral_data, (2, 0, 1))  # Convert to (H, W, B)
# # reconstruct RGB image from hyperspectral data
# spectral_data = spectral_data.astype(np.float32) / 65535.0
# spectral_data = (spectral_data - spectral_data.min()) / (spectral_data.max() - spectral_data.min())
# # print(f'Max value: {np.max(spectral_data)}, Min value: {np.min(spectral_data)}')
# red_band = spectral_data[:, :, 70] # the channel that is red
# green_band = spectral_data[:, :, 53] # the channel that is green
# blue_band = spectral_data[:, :, 19] # the channel that is blue
# reconstructed_rgb_img = np.stack([red_band, green_band, blue_band], axis=-1)

# reconstructed_rgb_img_flipped = np.fliplr(reconstructed_rgb_img)

# show_labeled_image(reconstructed_rgb_img_flipped, data_NZ2023_a_na['sorted_tensor'].iloc[0])

#         # Display image
#         # plt.imshow(rgb_image)
#         # plt.axis("off")
#         # plt.title("RGB Composite from Hyperspectral Data extracted from tensor")
#         # plt.show()

#         # Find tensors and sort from reconstructed RGB image



# In[ ]:


# Get a small dataset
# data_test = data_NZ2023_a_na[0:10]


# In[ ]:


# NZ2023_loc_link = 'spectral_NZ_2023'
# data_test['sorted_tensor'] = data_test['Image_folder_A'].apply(lambda x: extractor(x, model2, NZ2023_loc_link))


# In[ ]:


# pd.options.display.max_colwidth = None
# print(data_test['sorted_tensor'][])


# In[ ]:


# print(data_test)


# In[ ]:


# Visualise the masks detected vs the number of apples in the image
# folder_path = f'{folder_link}{UK2024_loc_link}{test_data["Spectral_folder_A"][0]}'

# png_file = glob.glob(f'{folder_path}/REFLECTANCE*.png')

# png_file = utils.read_image(png_file[0])

# plt.imshow(png_file)
# print(test_data['Image_folder_A'].value_counts())


# In[ ]:


# print(data_UK2024_a_na['Image_folder_A'][:6].value_counts())


# In[21]:


# Get df of missing images
# test_missing = test_data[test_data['tensors'].apply(is_empty_dict)]

missing_img_NZ2023_a = data_NZ2023_a_na[data_NZ2023_a_na['tensors'].apply(is_empty_dict)]
missing_img_NZ2023_b = data_NZ2023_b_na[data_NZ2023_b_na['tensors'].apply(is_empty_dict)]
missing_img_NZ2023_c = data_NZ2023_c_na[data_NZ2023_c_na['tensors'].apply(is_empty_dict)]
missing_img_NZ2023_d = data_NZ2023_d_na[data_NZ2023_d_na['tensors'].apply(is_empty_dict)]
missing_img_NZ2024_a = data_NZ2024_a_na[data_NZ2024_a_na['tensors'].apply(is_empty_dict)]
missing_img_NZ2024_b = data_NZ2024_b_na[data_NZ2024_b_na['tensors'].apply(is_empty_dict)]
missing_img_NZ2024_c = data_NZ2024_c_na[data_NZ2024_c_na['tensors'].apply(is_empty_dict)]
missing_img_NZ2024_d = data_NZ2024_d_na[data_NZ2024_d_na['tensors'].apply(is_empty_dict)]
missing_img_UK2024_a = data_UK2024_a_na[data_UK2024_a_na['tensors'].apply(is_empty_dict)]
missing_img_UK2024_b = data_UK2024_b_na[data_UK2024_b_na['tensors'].apply(is_empty_dict)]
missing_img_UK2024_c = data_UK2024_c_na[data_UK2024_c_na['tensors'].apply(is_empty_dict)]   
missing_img_UK2024_d = data_UK2024_d_na[data_UK2024_d_na['tensors'].apply(is_empty_dict)]


# In[ ]:


# # Test dataset
# len_test = len(test_missing)

# pickle_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Pickle_files/Tensors/'


# data = {
#     'List Name': [
#         'test_missing'
#     ],
#     'List Length': [
#         len_test
#     ]
# }

# df = pd.DataFrame(data)
# df.to_csv(f'{pickle_path}summary_test.csv', index=False)

# print("The lists and their respective lengths have been saved to lists_and_lengths.csv.")

# with open(f'{pickle_path}test_missing.pkl', 'wb') as f:
#     pickle.dump(test_missing, f)


# In[23]:


len_nz2023_a = len(missing_img_NZ2023_a)
len_nz2024_a = len(missing_img_NZ2024_a)
len_uk2024_a = len(missing_img_UK2024_a)
len_nz2023_b = len(missing_img_NZ2023_b)
len_nz2024_b = len(missing_img_NZ2024_b)
len_uk2024_b = len(missing_img_UK2024_b)
len_nz2023_c = len(missing_img_NZ2023_c)
len_nz2024_c = len(missing_img_NZ2024_c)
len_uk2024_c = len(missing_img_UK2024_c)
len_nz2023_d = len(missing_img_NZ2023_d)
len_nz2024_d = len(missing_img_NZ2024_d)
len_uk2024_d = len(missing_img_UK2024_d)

pickle_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Pickle_files/Tensors/'


data = {
    'List Name': [
        'missing_img_unique_nz2023_a', 'missing_img_unique_nz2024_a', 'missing_img_unique_uk2024_a',
        'missing_img_unique_nz2023_b', 'missing_img_unique_nz2024_b', 'missing_img_unique_uk2024_b',
        'missing_img_unique_nz2023_c', 'missing_img_unique_nz2024_c', 'missing_img_unique_uk2024_c',
        'missing_img_unique_nz2023_d', 'missing_img_unique_nz2024_d', 'missing_img_unique_uk2024_d'
    ],
    'List Length': [
        len_nz2023_a, len_nz2024_a, len_uk2024_a,
        len_nz2023_b, len_nz2024_b, len_uk2024_b,
        len_nz2023_c, len_nz2024_c, len_uk2024_c,
        len_nz2023_d, len_nz2024_d, len_uk2024_d
    ]
}


df = pd.DataFrame(data)
df.to_csv(f'{pickle_path}summary.csv', index=False)

# Save the missing lists to pickle files
save_if_not_empty('missing_img_NZ2023_a', missing_img_NZ2023_a)
save_if_not_empty('missing_img_NZ2023_b', missing_img_NZ2023_b)
save_if_not_empty('missing_img_NZ2023_c', missing_img_NZ2023_c)
save_if_not_empty('missing_img_NZ2023_d', missing_img_NZ2023_d)
save_if_not_empty('missing_img_NZ2024_a', missing_img_NZ2024_a)
save_if_not_empty('missing_img_NZ2024_b', missing_img_NZ2024_b)
save_if_not_empty('missing_img_NZ2024_c', missing_img_NZ2024_c)
save_if_not_empty('missing_img_NZ2024_d', missing_img_NZ2024_d)
save_if_not_empty('missing_img_UK2024_a', missing_img_UK2024_a)
save_if_not_empty('missing_img_UK2024_b', missing_img_UK2024_b)
save_if_not_empty('missing_img_UK2024_c', missing_img_UK2024_c)
save_if_not_empty('missing_img_UK2024_d', missing_img_UK2024_d)

print("The missing images have been saved to pickle files.")


# In[24]:


# # removing rows which miscount the number of apples in the image

# data_final_test = test_data[~test_data['tensors'].apply(is_empty_dict)]
# data_final_test = data_final_test.reset_index(drop=True)

data_NZ2023_a_final = data_NZ2023_a_na[~data_NZ2023_a_na['tensors'].apply(is_empty_dict)]
data_NZ2023_b_final = data_NZ2023_b_na[~data_NZ2023_b_na['tensors'].apply(is_empty_dict)]
data_NZ2023_c_final = data_NZ2023_c_na[~data_NZ2023_c_na['tensors'].apply(is_empty_dict)]
data_NZ2023_d_final = data_NZ2023_d_na[~data_NZ2023_d_na['tensors'].apply(is_empty_dict)]
# Reset the index
data_NZ2023_a_final = data_NZ2023_a_final.reset_index(drop=True)
data_NZ2023_b_final = data_NZ2023_b_final.reset_index(drop=True)
data_NZ2023_c_final = data_NZ2023_c_final.reset_index(drop=True)
data_NZ2023_d_final = data_NZ2023_d_final.reset_index(drop=True)


data_NZ2024_a_final = data_NZ2024_a_na[~data_NZ2024_a_na['tensors'].apply(is_empty_dict)]
data_NZ2024_b_final = data_NZ2024_b_na[~data_NZ2024_b_na['tensors'].apply(is_empty_dict)]
data_NZ2024_c_final = data_NZ2024_c_na[~data_NZ2024_c_na['tensors'].apply(is_empty_dict)]
data_NZ2024_d_final = data_NZ2024_d_na[~data_NZ2024_d_na['tensors'].apply(is_empty_dict)]
# Reset the index
data_NZ2024_a_final = data_NZ2024_a_final.reset_index(drop=True)
data_NZ2024_b_final = data_NZ2024_b_final.reset_index(drop=True)
data_NZ2024_c_final = data_NZ2024_c_final.reset_index(drop=True)
data_NZ2024_d_final = data_NZ2024_d_final.reset_index(drop=True)


data_UK2024_a_final = data_UK2024_a_na[~data_UK2024_a_na['tensors'].apply(is_empty_dict)]
data_UK2024_b_final = data_UK2024_b_na[~data_UK2024_b_na['tensors'].apply(is_empty_dict)]
data_UK2024_c_final = data_UK2024_c_na[~data_UK2024_c_na['tensors'].apply(is_empty_dict)]
data_UK2024_d_final = data_UK2024_d_na[~data_UK2024_d_na['tensors'].apply(is_empty_dict)]
# Reset the index
data_UK2024_a_final = data_UK2024_a_final.reset_index(drop=True)
data_UK2024_b_final = data_UK2024_b_final.reset_index(drop=True)
data_UK2024_c_final = data_UK2024_c_final.reset_index(drop=True)
data_UK2024_d_final = data_UK2024_d_final.reset_index(drop=True)


# In[25]:


# Save data_final to pickle
pickle_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Pickle_files/Tensors/'

# data_final_test.to_pickle(f'{pickle_path}data_final_test.pkl')

data_NZ2023_a_final.to_pickle(f'{pickle_path}data_NZ2023_a_final.pkl')
data_NZ2023_b_final.to_pickle(f'{pickle_path}data_NZ2023_b_final.pkl')
data_NZ2023_c_final.to_pickle(f'{pickle_path}data_NZ2023_c_final.pkl')
data_NZ2023_d_final.to_pickle(f'{pickle_path}data_NZ2023_d_final.pkl')
data_NZ2024_a_final.to_pickle(f'{pickle_path}data_NZ2024_a_final.pkl')
data_NZ2024_b_final.to_pickle(f'{pickle_path}data_NZ2024_b_final.pkl')
data_NZ2024_c_final.to_pickle(f'{pickle_path}data_NZ2024_c_final.pkl')
data_NZ2024_d_final.to_pickle(f'{pickle_path}data_NZ2024_d_final.pkl')
data_UK2024_a_final.to_pickle(f'{pickle_path}data_UK2024_a_final.pkl')
data_UK2024_b_final.to_pickle(f'{pickle_path}data_UK2024_b_final.pkl')
data_UK2024_c_final.to_pickle(f'{pickle_path}data_UK2024_c_final.pkl')
data_UK2024_d_final.to_pickle(f'{pickle_path}data_UK2024_d_final.pkl')


# In[85]:


# Load missing data
import pandas as pd
pickle_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Pickle_files/Tensors/'


missing_img_NZ2023_a = pd.read_pickle(f'{pickle_path}missing_img_NZ2023_a.pkl')
missing_img_NZ2023_b = pd.read_pickle(f'{pickle_path}missing_img_NZ2023_b.pkl')
missing_img_NZ2023_c = pd.read_pickle(f'{pickle_path}missing_img_NZ2023_c.pkl')
missing_img_NZ2023_d = pd.read_pickle(f'{pickle_path}missing_img_NZ2023_d.pkl')
missing_img_NZ2024_a = pd.read_pickle(f'{pickle_path}missing_img_NZ2024_a.pkl')
missing_img_NZ2024_b = pd.read_pickle(f'{pickle_path}missing_img_NZ2024_b.pkl')
missing_img_NZ2024_c = pd.read_pickle(f'{pickle_path}missing_img_NZ2024_c.pkl')
missing_img_NZ2024_d = pd.read_pickle(f'{pickle_path}missing_img_NZ2024_d.pkl')
missing_img_UK2024_a = pd.read_pickle(f'{pickle_path}missing_img_UK2024_a.pkl')
missing_img_UK2024_b = pd.read_pickle(f'{pickle_path}missing_img_UK2024_b.pkl')
missing_img_UK2024_c = pd.read_pickle(f'{pickle_path}missing_img_UK2024_c.pkl')
missing_img_UK2024_d = pd.read_pickle(f'{pickle_path}missing_img_UK2024_d.pkl')

file_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/csv_files/All_data_csv/'


data_NZ2023_a_na = pd.read_csv(f'{file_path}data_NZ2023_a_na.csv')
data_NZ2023_b_na = pd.read_csv(f'{file_path}data_NZ2023_b_na.csv')
data_NZ2023_c_na = pd.read_csv(f'{file_path}data_NZ2023_c_na.csv')
data_NZ2023_d_na = pd.read_csv(f'{file_path}data_NZ2023_d_na.csv')
data_NZ2024_a_na = pd.read_csv(f'{file_path}data_NZ2024_a_na.csv')
data_NZ2024_b_na = pd.read_csv(f'{file_path}data_NZ2024_b_na.csv')
data_NZ2024_c_na = pd.read_csv(f'{file_path}data_NZ2024_c_na.csv')
data_NZ2024_d_na = pd.read_csv(f'{file_path}data_NZ2024_d_na.csv')
data_UK2024_a_na = pd.read_csv(f'{file_path}data_UK2024_a_na.csv')
data_UK2024_b_na = pd.read_csv(f'{file_path}data_UK2024_b_na.csv')
data_UK2024_c_na = pd.read_csv(f'{file_path}data_UK2024_c_na.csv')
data_UK2024_d_na = pd.read_csv(f'{file_path}data_UK2024_d_na.csv')


# In[88]:


# Calculate the number of missing apples per file

NZ2023_a_perc = (len(missing_img_NZ2023_a)/len(data_NZ2023_a_na))*100
NZ2023_b_perc = (len(missing_img_NZ2023_b)/len(data_NZ2023_b_na))*100
NZ2023_c_perc = (len(missing_img_NZ2023_c)/len(data_NZ2023_c_na))*100
NZ2023_d_perc = (len(missing_img_NZ2023_d)/len(data_NZ2023_d_na))*100
NZ2024_a_perc = (len(missing_img_NZ2024_a)/len(data_NZ2024_a_na))*100
NZ2024_b_perc = (len(missing_img_NZ2024_b)/len(data_NZ2024_b_na))*100
NZ2024_c_perc = (len(missing_img_NZ2024_c)/len(data_NZ2024_c_na))*100
NZ2024_d_perc = (len(missing_img_NZ2024_d)/len(data_NZ2024_d_na))*100
UK2024_a_perc = (len(missing_img_UK2024_a)/len(data_UK2024_a_na))*100
UK2024_b_perc = (len(missing_img_UK2024_b)/len(data_UK2024_b_na))*100
UK2024_c_perc = (len(missing_img_UK2024_c)/len(data_UK2024_c_na))*100
UK2024_d_perc = (len(missing_img_UK2024_d)/len(data_UK2024_d_na))*100


# In[89]:


print(f'NZ2023_a: {NZ2023_a_perc}')
print(f'NZ2023_b: {NZ2023_b_perc}')
print(f'NZ2023_c: {NZ2023_c_perc}')
print(f'NZ2023_d: {NZ2023_d_perc}')
print(f'NZ2024_a: {NZ2024_a_perc}') 
print(f'NZ2024_b: {NZ2024_b_perc}')
print(f'NZ2024_c: {NZ2024_c_perc}')
print(f'NZ2024_d: {NZ2024_d_perc}')
print(f'UK2024_a: {UK2024_a_perc}')
print(f'UK2024_b: {UK2024_b_perc}')
print(f'UK2024_c: {UK2024_c_perc}')
print(f'UK2024_d: {UK2024_d_perc}')


# In[101]:


missing_img_NZ2023_a = missing_img_NZ2023_a.reset_index(drop=True)
missing_img_NZ2023_b = missing_img_NZ2023_b.reset_index(drop=True)
missing_img_NZ2023_c = missing_img_NZ2023_c.reset_index(drop=True)
missing_img_NZ2023_d = missing_img_NZ2023_d.reset_index(drop=True)
missing_img_NZ2024_a = missing_img_NZ2024_a.reset_index(drop=True)
missing_img_NZ2024_b = missing_img_NZ2024_b.reset_index(drop=True)
missing_img_NZ2024_c = missing_img_NZ2024_c.reset_index(drop=True)
missing_img_NZ2024_d = missing_img_NZ2024_d.reset_index(drop=True)
missing_img_UK2024_a = missing_img_UK2024_a.reset_index(drop=True)
missing_img_UK2024_b = missing_img_UK2024_b.reset_index(drop=True)
missing_img_UK2024_c = missing_img_UK2024_c.reset_index(drop=True)
missing_img_UK2024_d = missing_img_UK2024_d.reset_index(drop=True)


# In[ ]:


# Create unique dataframes for each side
missing_img_NZ2023_a_unique = missing_img_NZ2023_a.drop_duplicates(subset=['Spectral_folder_A'])
missing_img_NZ2023_b_unique = missing_img_NZ2023_b.drop_duplicates(subset=['Spectral_folder_B'])
missing_img_NZ2023_c_unique = missing_img_NZ2023_c.drop_duplicates(subset=['Spectral_folder_C'])
missing_img_NZ2023_d_unique = missing_img_NZ2023_d.drop_duplicates(subset=['Spectral_folder_D'])
missing_img_NZ2024_a_unique = missing_img_NZ2024_a.drop_duplicates(subset=['Spectral_folder_A'])
missing_img_NZ2024_b_unique = missing_img_NZ2024_b.drop_duplicates(subset=['Spectral_folder_B'])
missing_img_NZ2024_c_unique = missing_img_NZ2024_c.drop_duplicates(subset=['Spectral_folder_C'])
missing_img_NZ2024_d_unique = missing_img_NZ2024_d.drop_duplicates(subset=['Spectral_folder_D'])
missing_img_UK2024_a_unique = missing_img_UK2024_a.drop_duplicates(subset=['Spectral_folder_A'])
missing_img_UK2024_b_unique = missing_img_UK2024_b.drop_duplicates(subset=['Spectral_folder_B'])
missing_img_UK2024_c_unique = missing_img_UK2024_c.drop_duplicates(subset=['Spectral_folder_C'])
missing_img_UK2024_d_unique = missing_img_UK2024_d.drop_duplicates(subset=['Spectral_folder_D'])


# In[ ]:


# data_final_test2 = pd.read_pickle(f'{pickle_path}data_final_test.pkl')

# print(data_final_test2)


# In[ ]:


# print(data_NZ2023_a_final2.describe())


# In[ ]:


# # load data_final to pickle
# pickle_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Pickle_files/'

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


# In[ ]:


# min_starch = data_NZ2023_d_final2['Starch'].min()
# max_starch = data_NZ2023_d_final2['Starch'].max()

# # Print the range
# print(f"The range of Starch values is: {min_starch} to {max_starch}")


# ## Attempting to get more tensors extracted from the ones unable to generate previously ##

# In[ ]:


mask_generator_3 = SAM2AutomaticMaskGenerator(

    model=sam2,
    points_per_side=40,
    points_per_batch=50,
    pred_iou_thresh=0.85,
    stability_score_thresh=0.9,
    stability_score_offset=0.85,
    crop_n_layers=1,
    box_nms_thresh=0, # low values for non-overlapping bboxes
    crop_n_points_downscale_factor=8,
    min_mask_region_area=8000.0,
    use_m2m=True,
)


# In[99]:


# Using SAM segregation and filters to identify correct masks
def extractor_mask_gen3(folder_link, loc_link, row, side, intensity_threshold=40, circularity_threshold=0.6):
    """
    Extract sorted, filtered SAM masks from an image in a specified folder.

    Args:
        folder_link (str): Base folder path.
        loc_link (str): Subdirectory path.
        folder_name (str): Folder name where image is stored.
        intensity_threshold (int): Minimum intensity to consider foreground.
        circularity_threshold (float): Circularity threshold for mask filtering.

    Returns:
        list: Sorted and filtered list of masks.
    """
     
    side = side 

    # Count = row['Count']
    folder_name = row[f'Spectral_folder_{side}']

    folder_path = f'{folder_link}{loc_link}{folder_name}'
    png_file = glob.glob(f'{folder_path}/REFLECTANCE*.png')
    if not png_file:
        return []
    
    rgb_image = utils.read_image(png_file[0])
    gen_mask = mask_generator_3.generate(rgb_image)
  

    filtered_masks = []

    for mask_data in gen_mask:
        mask = mask_data['segmentation']
        bbox = mask_data['bbox']
        area = mask_data['area']
        width, height = bbox[2], bbox[3]
            # Intensity check
        if rgb_image[mask.astype(bool)].max() <= intensity_threshold:
            # plot_individual_masks(mask, rgb_image)
            continue
        
        # Shape check
        if width > 1.5 * height:
            continue
        
        # Area check
        if area <= 5000:
            continue
        
        # Circularity check
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (contour_area / (perimeter ** 2))
        if circularity < circularity_threshold:
            continue
        
        filtered_masks.append(mask_data)
    if not filtered_masks:
        return []
    filtered_masks_no_base = remove_lower_overlapping_boxes(filtered_masks)
    filtered_masks_stalkless = process_stalks(filtered_masks_no_base, top_fraction=0.2)

    sorted_masks = reorganize_masks(filtered_masks_stalkless)
    

    return sorted_masks


# In[97]:


# Function to use extract function and return tensors in the correct format
def extractor_spectral2(df_unique, folder_link, loc_link, side):
    """
    Extract tensors into each row of the dataframe
    Args:
        df_unique (DataFrame): DataFrame with unique rows.
        folder_link (str): Base folder path.
        loc_link (str): Subdirectory path.
        side (str): Side of the image (A, B, C, D).
    Returns:
        a pd.Series: Series of tensors to be combined into a DataFrame
    """
    
    # Apply the extractor function to each row
    df_unique['sorted_tensor'] = df_unique.apply(
        lambda row: extractor_mask_gen3(folder_link, loc_link, row, side=side), axis=1)
    
    
    tensor_list = []
    total_length = len(df_unique)
    row_num = 1

    # Iterate for each row in the DataFrame and update tensor list
    for index, row in df_unique.iterrows():
        
        count = row['Count']
        sorted_tensors = row['sorted_tensor']

        if len(sorted_tensors) == count:
            print(f"Row {index}: Count matches length of sorted_tensor ({count})")
            tensor_list.append(sorted_tensors)
        else:
            print(f"Row {index}: Count does not match length of sorted_tensor (Count: {count}, Length: {len(sorted_tensors)})")
            tensor_list.append([{} for _ in range(count)])  # Append an empty dictionary if they don't match by the number of counts

        print(f"Completed {row_num}/{total_length} rows")
        row_num += 1

    flattened_list = [item for sublist in tensor_list for item in sublist]
    # tensor_list_pd = pd.Series(flattened_list)
    return flattened_list

      
    


# In[109]:


test = missing_img_NZ2023_a [:6].reset_index(drop=True)


# In[110]:


print(test)


# In[ ]:





# In[ ]:


folder_link = "/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/"

intensity_threshold=40 
circularity_threshold=0.6


NZ2023_loc_link = 'spectral_NZ_2023/'
NZ2024_loc_link = 'spectral_NZ_2024/'
UK2024_loc_link = 'spectral_UK_2024/'

side = 'A' 

# Count = row['Count']
folder_name = test[f'Spectral_folder_{side}'][0]
print(NZ2023_loc_link)
print(folder_name)

folder_path = f'{folder_link}{NZ2023_loc_link}{folder_name}'
print(folder_path)
png_file = glob.glob(f'{folder_path}/REFLECTANCE*.png')
print(png_file)
if not png_file:
    print(f"No PNG file found in {folder_path}")

rgb_image = utils.read_image(png_file[0])


# In[ ]:





# In[82]:


gen_mask = mask_generator_3.generate(rgb_image)


# In[90]:


# fig, ax = plt.subplots(1, 3, figsize=(10, 3))

# for i, row in test.iterrows():

# print(f"gen_mask count:{len(gen_mask)}")
# show_labeled_image(rgb_image, row['tensor'].iloc[0])
plt.imshow(rgb_image)
show_anns(gen_mask)
plt.axis('off')
plt.show()
print(len(gen_mask))
filtered_masks = []

# ax[i].imshow(rgb_image)

for mask_data in gen_mask:
    mask = mask_data['segmentation']
    bbox = mask_data['bbox']
    area = mask_data['area']
    width, height = bbox[2], bbox[3]
    


        # Intensity check
    if rgb_image[mask.astype(bool)].max() <= intensity_threshold:
        continue
    
    # Shape check
    if width > 1.5 * height:
        continue

    # Area check
    if area <= 5000:
        continue

    # Circularity check
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue
    contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        continue
    circularity = 4 * np.pi * (contour_area / (perimeter ** 2))
    if circularity < circularity_threshold:
        continue

    filtered_masks.append(mask_data)
if not filtered_masks:    
    print("No filtered masks found")

filtered_masks_no_base = remove_lower_overlapping_boxes(filtered_masks)
filtered_masks_stalkless = process_stalks(filtered_masks_no_base, top_fraction=0.2)

sorted_masks = reorganize_masks(filtered_masks_stalkless)

plt.imshow(rgb_image)
show_anns(sorted_masks)

print(len(sorted_masks))


# In[104]:


print(missing_img_NZ2023_a[:9])


# In[114]:


print(missing_img_NZ2023_a_unique[:1])


# In[118]:


test['tensors'] = extractor_spectral2(missing_img_NZ2023_a_unique[:1], folder_link, NZ2023_loc_link, 'A')


# In[119]:


print(len(missing_img_NZ2023_a_unique))
print(len(missing_img_NZ2023_b_unique))
print(len(missing_img_NZ2023_c_unique))
print(len(missing_img_NZ2023_d_unique))
print(len(missing_img_NZ2024_a_unique))
print(len(missing_img_NZ2024_b_unique))
print(len(missing_img_NZ2024_c_unique))
print(len(missing_img_NZ2024_d_unique))
print(len(missing_img_UK2024_a_unique))
print(len(missing_img_UK2024_b_unique))
print(len(missing_img_UK2024_c_unique))
print(len(missing_img_UK2024_d_unique))


# In[122]:


# Extractor spectral2 uses a less stringent mask generator
folder_link = "/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/"

NZ2023_loc_link = 'spectral_NZ_2023/'
NZ2024_loc_link = 'spectral_NZ_2024/'
UK2024_loc_link = 'spectral_UK_2024/'

# print(test_data)

# test_data['tensors'] = extractor_spectral(test_data_unique, folder_link, UK2024_loc_link, side = 'A')


missing_img_NZ2023_a['tensors'] = extractor_spectral2(missing_img_NZ2023_a_unique, folder_link, NZ2023_loc_link, 'A')
missing_img_NZ2023_b['tensors'] = extractor_spectral2(missing_img_NZ2023_b_unique, folder_link, NZ2023_loc_link, 'B')
missing_img_NZ2023_c['tensors'] = extractor_spectral2(missing_img_NZ2023_c_unique, folder_link, NZ2023_loc_link, 'C') 
missing_img_NZ2023_d['tensors'] = extractor_spectral2(missing_img_NZ2023_d_unique, folder_link, NZ2023_loc_link, 'D')
# data_NZ2023_a_na['sorted_tensor'].head(5)


# In[130]:


missing_img_NZ2023_a2 = missing_img_NZ2023_a[missing_img_NZ2023_a['tensors'].apply(is_empty_dict)]
missing_img_NZ2023_b2 = missing_img_NZ2023_b[missing_img_NZ2023_b['tensors'].apply(is_empty_dict)]
missing_img_NZ2023_c2 = missing_img_NZ2023_c[missing_img_NZ2023_c['tensors'].apply(is_empty_dict)]
missing_img_NZ2023_d2 = missing_img_NZ2023_d[missing_img_NZ2023_d['tensors'].apply(is_empty_dict)]

save_if_not_empty('missing_img_NZ2023_a2', missing_img_NZ2023_a2)
save_if_not_empty('missing_img_NZ2023_b2', missing_img_NZ2023_b2)
save_if_not_empty('missing_img_NZ2023_c2', missing_img_NZ2023_c2)
save_if_not_empty('missing_img_NZ2023_d2', missing_img_NZ2023_d2)


data_NZ2023_a_final2 = missing_img_NZ2023_a2[~missing_img_NZ2023_a2['tensors'].apply(is_empty_dict)]
data_NZ2023_b_final2 = missing_img_NZ2023_b2[~missing_img_NZ2023_b2['tensors'].apply(is_empty_dict)]
data_NZ2023_c_final2 = missing_img_NZ2023_c2[~missing_img_NZ2023_c2['tensors'].apply(is_empty_dict)]
data_NZ2023_d_final2 = missing_img_NZ2023_d2[~missing_img_NZ2023_d2['tensors'].apply(is_empty_dict)]
# Reset the index
data_NZ2023_a_final2 = data_NZ2023_a_final2.reset_index(drop=True)
data_NZ2023_b_final2 = data_NZ2023_b_final2.reset_index(drop=True)
data_NZ2023_c_final2 = data_NZ2023_c_final2.reset_index(drop=True)
data_NZ2023_d_final2 = data_NZ2023_d_final2.reset_index(drop=True)



# In[131]:


missing_img_NZ2024_a['tensors'] = extractor_spectral2(missing_img_NZ2024_a_unique, folder_link, NZ2024_loc_link, 'A')
missing_img_NZ2024_b['tensors'] = extractor_spectral2(missing_img_NZ2024_b_unique, folder_link, NZ2024_loc_link, 'B')
missing_img_NZ2024_c['tensors'] = extractor_spectral2(missing_img_NZ2024_c_unique, folder_link, NZ2024_loc_link, 'C') 
missing_img_NZ2024_d['tensors'] = extractor_spectral2(missing_img_NZ2024_d_unique, folder_link, NZ2024_loc_link, 'D')
# data_NZ2024_a_na['sorted_tensor'].head(5)



missing_img_UK2024_a['tensors'] = extractor_spectral2(missing_img_UK2024_a_unique, folder_link, UK2024_loc_link, 'A')
missing_img_UK2024_b['tensors'] = extractor_spectral2(missing_img_UK2024_b_unique, folder_link, UK2024_loc_link, 'B')
missing_img_UK2024_c['tensors'] = extractor_spectral2(missing_img_UK2024_c_unique, folder_link, UK2024_loc_link, 'C') 
missing_img_UK2024_d['tensors'] = extractor_spectral2(missing_img_UK2024_d_unique, folder_link, UK2024_loc_link, 'D')


# In[132]:


# Get df of missing images
# test_missing = test_data[test_data['tensors'].apply(is_empty_dict)]


missing_img_NZ2024_a2 = missing_img_NZ2024_a[missing_img_NZ2024_a['tensors'].apply(is_empty_dict)]
missing_img_NZ2024_b2 = missing_img_NZ2024_b[missing_img_NZ2024_b['tensors'].apply(is_empty_dict)]
missing_img_NZ2024_c2 = missing_img_NZ2024_c[missing_img_NZ2024_c['tensors'].apply(is_empty_dict)]
missing_img_NZ2024_d2 = missing_img_NZ2024_d[missing_img_NZ2024_d['tensors'].apply(is_empty_dict)]
missing_img_UK2024_a2 = missing_img_UK2024_a[missing_img_UK2024_a['tensors'].apply(is_empty_dict)]
missing_img_UK2024_b2 = missing_img_UK2024_b[missing_img_UK2024_b['tensors'].apply(is_empty_dict)]
missing_img_UK2024_c2 = missing_img_UK2024_c[missing_img_UK2024_c['tensors'].apply(is_empty_dict)]   
missing_img_UK2024_d2 = missing_img_UK2024_d[missing_img_UK2024_d['tensors'].apply(is_empty_dict)]


# In[133]:


len_nz2023_a = len(missing_img_NZ2023_a2)
len_nz2024_a = len(missing_img_NZ2024_a2)
len_uk2024_a = len(missing_img_UK2024_a2)
len_nz2023_b = len(missing_img_NZ2023_b2)
len_nz2024_b = len(missing_img_NZ2024_b2)
len_uk2024_b = len(missing_img_UK2024_b2)
len_nz2023_c = len(missing_img_NZ2023_c2)
len_nz2024_c = len(missing_img_NZ2024_c2)
len_uk2024_c = len(missing_img_UK2024_c2)
len_nz2023_d = len(missing_img_NZ2023_d2)
len_nz2024_d = len(missing_img_NZ2024_d2)
len_uk2024_d = len(missing_img_UK2024_d2)

pickle_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Pickle_files/Tensors/'


data = {
    'List Name': [
        'missing_img_unique_nz2023_a2', 'missing_img_unique_nz2024_a2', 'missing_img_unique_uk2024_a2',
        'missing_img_unique_nz2023_b2', 'missing_img_unique_nz2024_b2', 'missing_img_unique_uk2024_b2',
        'missing_img_unique_nz2023_c2', 'missing_img_unique_nz2024_c2', 'missing_img_unique_uk2024_c2',
        'missing_img_unique_nz2023_d2', 'missing_img_unique_nz2024_d2', 'missing_img_unique_uk2024_d2'
    ],
    'List Length': [
        len_nz2023_a, len_nz2024_a, len_uk2024_a,
        len_nz2023_b, len_nz2024_b, len_uk2024_b,
        len_nz2023_c, len_nz2024_c, len_uk2024_c,
        len_nz2023_d, len_nz2024_d, len_uk2024_d
    ]
}


df = pd.DataFrame(data)
df.to_csv(f'{pickle_path}summary2.csv', index=False)

# Save the missing lists to pickle files

save_if_not_empty('missing_img_NZ2024_a2', missing_img_NZ2024_a2)
save_if_not_empty('missing_img_NZ2024_b2', missing_img_NZ2024_b2)
save_if_not_empty('missing_img_NZ2024_c2', missing_img_NZ2024_c2)
save_if_not_empty('missing_img_NZ2024_d2', missing_img_NZ2024_d2)
save_if_not_empty('missing_img_UK2024_a2', missing_img_UK2024_a2)
save_if_not_empty('missing_img_UK2024_b2', missing_img_UK2024_b2)
save_if_not_empty('missing_img_UK2024_c2', missing_img_UK2024_c2)
save_if_not_empty('missing_img_UK2024_d2', missing_img_UK2024_d2)

print("The missing images have been saved to pickle files.")


# In[134]:


# # removing rows which miscount the number of apples in the image

# data_final_test = test_data[~test_data['tensors'].apply(is_empty_dict)]
# data_final_test = data_final_test.reset_index(drop=True)

data_NZ2024_a_final2 = missing_img_NZ2024_a2[~missing_img_NZ2024_a2['tensors'].apply(is_empty_dict)]
data_NZ2024_b_final2 = missing_img_NZ2024_b2[~missing_img_NZ2024_b2['tensors'].apply(is_empty_dict)]
data_NZ2024_c_final2 = missing_img_NZ2024_c2[~missing_img_NZ2024_c2['tensors'].apply(is_empty_dict)]
data_NZ2024_d_final2 = missing_img_NZ2024_d2[~missing_img_NZ2024_d2['tensors'].apply(is_empty_dict)]
# Reset the index
data_NZ2024_a_final2 = data_NZ2024_a_final2.reset_index(drop=True)
data_NZ2024_b_final2 = data_NZ2024_b_final2.reset_index(drop=True)
data_NZ2024_c_final2 = data_NZ2024_c_final2.reset_index(drop=True)
data_NZ2024_d_final2 = data_NZ2024_d_final2.reset_index(drop=True)


data_UK2024_a_final2 = missing_img_UK2024_a2[~missing_img_UK2024_a2['tensors'].apply(is_empty_dict)]
data_UK2024_b_final2 = missing_img_UK2024_b2[~missing_img_UK2024_b2['tensors'].apply(is_empty_dict)]
data_UK2024_c_final2 = missing_img_UK2024_c2[~missing_img_UK2024_c2['tensors'].apply(is_empty_dict)]
data_UK2024_d_final2 = missing_img_UK2024_d2[~missing_img_UK2024_d2['tensors'].apply(is_empty_dict)]
# Reset the index
data_UK2024_a_final2 = data_UK2024_a_final2.reset_index(drop=True)
data_UK2024_b_final2 = data_UK2024_b_final2.reset_index(drop=True)
data_UK2024_c_final2 = data_UK2024_c_final2.reset_index(drop=True)
data_UK2024_d_final2 = data_UK2024_d_final2.reset_index(drop=True)


# In[135]:


# Save data_final to pickle
pickle_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Pickle_files/Tensors/'

# data_final_test.to_pickle(f'{pickle_path}data_final_test.pkl')

data_NZ2023_a_final2.to_pickle(f'{pickle_path}data_NZ2023_a_final2.pkl')
data_NZ2023_b_final2.to_pickle(f'{pickle_path}data_NZ2023_b_final2.pkl')
data_NZ2023_c_final2.to_pickle(f'{pickle_path}data_NZ2023_c_final2.pkl')
data_NZ2023_d_final2.to_pickle(f'{pickle_path}data_NZ2023_d_final2.pkl')
data_NZ2024_a_final2.to_pickle(f'{pickle_path}data_NZ2024_a_final2.pkl')
data_NZ2024_b_final2.to_pickle(f'{pickle_path}data_NZ2024_b_final2.pkl')
data_NZ2024_c_final2.to_pickle(f'{pickle_path}data_NZ2024_c_final2.pkl')
data_NZ2024_d_final2.to_pickle(f'{pickle_path}data_NZ2024_d_final2.pkl')
data_UK2024_a_final2.to_pickle(f'{pickle_path}data_UK2024_a_final2.pkl')
data_UK2024_b_final2.to_pickle(f'{pickle_path}data_UK2024_b_final2.pkl')
data_UK2024_c_final2.to_pickle(f'{pickle_path}data_UK2024_c_final2.pkl')
data_UK2024_d_final2.to_pickle(f'{pickle_path}data_UK2024_d_final2.pkl')


# In[138]:


still_miss_NZ2023_a = len(missing_img_NZ2023_a) - len_nz2023_a
still_miss_NZ2023_b = len(missing_img_NZ2023_b) - len_nz2024_a
still_miss_NZ2023_c = len(missing_img_NZ2023_c) - len_uk2024_a
still_miss_NZ2023_d = len(missing_img_NZ2023_d) - len_nz2023_b
still_miss_NZ2024_a = len(missing_img_NZ2024_a) - len_nz2024_b
still_miss_NZ2024_b = len(missing_img_NZ2024_b) - len_uk2024_b
still_miss_NZ2024_c = len(missing_img_NZ2024_c) - len_nz2023_c
still_miss_NZ2024_d = len(missing_img_NZ2024_d) - len_nz2024_c
still_miss_UK2024_a = len(missing_img_UK2024_a) - len_uk2024_c
still_miss_UK2024_b = len(missing_img_UK2024_b) - len_nz2023_d
still_miss_UK2024_c = len(missing_img_UK2024_c) - len_nz2024_d
still_miss_UK2024_d = len(missing_img_UK2024_d) - len_uk2024_d

print(f'original: {len(missing_img_NZ2023_a)}, remaining:{still_miss_NZ2023_a}')
print(f'original: {len(missing_img_NZ2023_b)}, remaining:{still_miss_NZ2023_b}')
print(f'original: {len(missing_img_NZ2023_c)}, remaining:{still_miss_NZ2023_c}')
print(f'original: {len(missing_img_NZ2023_d)}, remaining:{still_miss_NZ2023_d}')
print(f'original: {len(missing_img_NZ2024_a)}, remaining:{still_miss_NZ2024_a}')
print(f'original: {len(missing_img_NZ2024_b)}, remaining:{still_miss_NZ2024_b}')
print(f'original: {len(missing_img_NZ2024_c)}, remaining:{still_miss_NZ2024_c}')
print(f'original: {len(missing_img_NZ2024_d)}, remaining:{still_miss_NZ2024_d}')
print(f'original: {len(missing_img_UK2024_a)}, remaining:{still_miss_UK2024_a}')
print(f'original: {len(missing_img_UK2024_b)}, remaining:{still_miss_UK2024_b}')
print(f'original: {len(missing_img_UK2024_c)}, remaining:{still_miss_UK2024_c}')
print(f'original: {len(missing_img_UK2024_d)}, remaining:{still_miss_UK2024_d}')









# ### Getting the remaining missing tensors ###

# In[156]:


mask_generator_4 = SAM2AutomaticMaskGenerator(

    model=sam2,
    points_per_side=52,
    points_per_batch=70,
    pred_iou_thresh=0.8,
    stability_score_thresh=0.7,
    stability_score_offset=0.7,
    crop_n_layers=1,
    box_nms_thresh=0, # low values for non-overlapping bboxes
    crop_n_points_downscale_factor=7,
    min_mask_region_area=8000.0,
    use_m2m=True,
)


# In[141]:


# Using SAM segregation and filters to identify correct masks
def extractor_mask_gen4(folder_link, loc_link, row, side, intensity_threshold=40, circularity_threshold=0.6):
    """
    Extract sorted, filtered SAM masks from an image in a specified folder.

    Args:
        folder_link (str): Base folder path.
        loc_link (str): Subdirectory path.
        folder_name (str): Folder name where image is stored.
        intensity_threshold (int): Minimum intensity to consider foreground.
        circularity_threshold (float): Circularity threshold for mask filtering.

    Returns:
        list: Sorted and filtered list of masks.
    """
     
    side = side 

    # Count = row['Count']
    folder_name = row[f'Spectral_folder_{side}']

    folder_path = f'{folder_link}{loc_link}{folder_name}'
    png_file = glob.glob(f'{folder_path}/REFLECTANCE*.png')
    if not png_file:
        return []
    
    rgb_image = utils.read_image(png_file[0])
    gen_mask = mask_generator_4.generate(rgb_image)
  

    filtered_masks = []

    for mask_data in gen_mask:
        mask = mask_data['segmentation']
        bbox = mask_data['bbox']
        area = mask_data['area']
        width, height = bbox[2], bbox[3]
            # Intensity check
        if rgb_image[mask.astype(bool)].max() <= intensity_threshold:
            # plot_individual_masks(mask, rgb_image)
            continue
        
        # Shape check
        if width > 1.5 * height:
            continue
        
        # Area check
        if area <= 5000:
            continue
        
        # Circularity check
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (contour_area / (perimeter ** 2))
        if circularity < circularity_threshold:
            continue
        
        filtered_masks.append(mask_data)
    if not filtered_masks:
        return []
    filtered_masks_no_base = remove_lower_overlapping_boxes(filtered_masks)
    filtered_masks_stalkless = process_stalks(filtered_masks_no_base, top_fraction=0.2)

    sorted_masks = reorganize_masks(filtered_masks_stalkless)
    

    return sorted_masks


# In[142]:


# Function to use extract function and return tensors in the correct format
def extractor_spectral3(df_unique, folder_link, loc_link, side):
    """
    Extract tensors into each row of the dataframe
    Args:
        df_unique (DataFrame): DataFrame with unique rows.
        folder_link (str): Base folder path.
        loc_link (str): Subdirectory path.
        side (str): Side of the image (A, B, C, D).
    Returns:
        a pd.Series: Series of tensors to be combined into a DataFrame
    """
    
    # Apply the extractor function to each row
    df_unique['sorted_tensor'] = df_unique.apply(
        lambda row: extractor_mask_gen4(folder_link, loc_link, row, side=side), axis=1)
    
    
    tensor_list = []
    total_length = len(df_unique)
    row_num = 1

    # Iterate for each row in the DataFrame and update tensor list
    for index, row in df_unique.iterrows():
        
        count = row['Count']
        sorted_tensors = row['sorted_tensor']

        if len(sorted_tensors) == count:
            print(f"Row {index}: Count matches length of sorted_tensor ({count})")
            tensor_list.append(sorted_tensors)
        else:
            print(f"Row {index}: Count does not match length of sorted_tensor (Count: {count}, Length: {len(sorted_tensors)})")
            tensor_list.append([{} for _ in range(count)])  # Append an empty dictionary if they don't match by the number of counts

        print(f"Completed {row_num}/{total_length} rows")
        row_num += 1

    flattened_list = [item for sublist in tensor_list for item in sublist]
    # tensor_list_pd = pd.Series(flattened_list)
    return flattened_list

      
    


# In[146]:


test = missing_img_NZ2023_b[:6].reset_index(drop=True)

print(test)


# In[157]:


folder_link = "/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/"

intensity_threshold=40 
circularity_threshold=0.6


NZ2023_loc_link = 'spectral_NZ_2023/'
NZ2024_loc_link = 'spectral_NZ_2024/'
UK2024_loc_link = 'spectral_UK_2024/'

side = 'B' 

# Count = row['Count']
folder_name = test[f'Spectral_folder_{side}'][0]

folder_path = f'{folder_link}{NZ2023_loc_link}{folder_name}'
print(folder_path)
png_file = glob.glob(f'{folder_path}/REFLECTANCE*.png')
print(png_file)
if not png_file:
    print(f"No PNG file found in {folder_path}")

rgb_image = utils.read_image(png_file[0])


# In[158]:


gen_mask = mask_generator_4.generate(rgb_image)


# In[159]:


# fig, ax = plt.subplots(1, 3, figsize=(10, 3))

# for i, row in test.iterrows():

# print(f"gen_mask count:{len(gen_mask)}")
# show_labeled_image(rgb_image, row['tensor'].iloc[0])
plt.imshow(rgb_image)
show_anns(gen_mask)
plt.axis('off')
plt.show()
print(len(gen_mask))
filtered_masks = []

# ax[i].imshow(rgb_image)

for mask_data in gen_mask:
    mask = mask_data['segmentation']
    bbox = mask_data['bbox']
    area = mask_data['area']
    width, height = bbox[2], bbox[3]
    


        # Intensity check
    if rgb_image[mask.astype(bool)].max() <= intensity_threshold:
        continue
    
    # Shape check
    if width > 1.5 * height:
        continue

    # Area check
    if area <= 5000:
        continue

    # Circularity check
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue
    contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        continue
    circularity = 4 * np.pi * (contour_area / (perimeter ** 2))
    if circularity < circularity_threshold:
        continue

    filtered_masks.append(mask_data)
if not filtered_masks:    
    print("No filtered masks found")

filtered_masks_no_base = remove_lower_overlapping_boxes(filtered_masks)
filtered_masks_stalkless = process_stalks(filtered_masks_no_base, top_fraction=0.2)

sorted_masks = reorganize_masks(filtered_masks_stalkless)

plt.imshow(rgb_image)
show_anns(sorted_masks)

print(len(sorted_masks))


# In[163]:


print(missing_img_NZ2023_a2[:3])


# In[164]:


missing_img_NZ2023_a2_unique = missing_img_NZ2023_a2.drop_duplicates(subset=['Spectral_folder_A'])
missing_img_NZ2023_b2_unique = missing_img_NZ2023_b2.drop_duplicates(subset=['Spectral_folder_B'])
missing_img_NZ2023_c2_unique = missing_img_NZ2023_c2.drop_duplicates(subset=['Spectral_folder_C'])
missing_img_NZ2023_d2_unique = missing_img_NZ2023_d2.drop_duplicates(subset=['Spectral_folder_D'])


# In[165]:


# Extractor spectral2 uses a less stringent mask generator
folder_link = "/media/2tbdisk2/data/Haidee_apple_data/Haidee/Hyperspectral_images/"

NZ2023_loc_link = 'spectral_NZ_2023/'
NZ2024_loc_link = 'spectral_NZ_2024/'
UK2024_loc_link = 'spectral_UK_2024/'

# print(test_data)

# test_data['tensors'] = extractor_spectral(test_data_unique, folder_link, UK2024_loc_link, side = 'A')


missing_img_NZ2023_a2['tensors'] = extractor_spectral3(missing_img_NZ2023_a2_unique, folder_link, NZ2023_loc_link, 'A')
missing_img_NZ2023_b2['tensors'] = extractor_spectral3(missing_img_NZ2023_b2_unique, folder_link, NZ2023_loc_link, 'B')
missing_img_NZ2023_c2['tensors'] = extractor_spectral3(missing_img_NZ2023_c2_unique, folder_link, NZ2023_loc_link, 'C') 
missing_img_NZ2023_d2['tensors'] = extractor_spectral3(missing_img_NZ2023_d2_unique, folder_link, NZ2023_loc_link, 'D')
# data_NZ2023_a_na['sorted_tensor'].head(5)


# In[166]:


missing_img_NZ2023_a3 = missing_img_NZ2023_a2[missing_img_NZ2023_a2['tensors'].apply(is_empty_dict)]
missing_img_NZ2023_b3 = missing_img_NZ2023_b2[missing_img_NZ2023_b2['tensors'].apply(is_empty_dict)]
missing_img_NZ2023_c3 = missing_img_NZ2023_c2[missing_img_NZ2023_c2['tensors'].apply(is_empty_dict)]
missing_img_NZ2023_d3 = missing_img_NZ2023_d2[missing_img_NZ2023_d2['tensors'].apply(is_empty_dict)]

save_if_not_empty('missing_img_NZ2023_a3', missing_img_NZ2023_a3)
save_if_not_empty('missing_img_NZ2023_b3', missing_img_NZ2023_b3)
save_if_not_empty('missing_img_NZ2023_c3', missing_img_NZ2023_c3)
save_if_not_empty('missing_img_NZ2023_d3', missing_img_NZ2023_d3)


data_NZ2023_a_final3 = missing_img_NZ2023_a2[~missing_img_NZ2023_a2['tensors'].apply(is_empty_dict)]
data_NZ2023_b_final3 = missing_img_NZ2023_b2[~missing_img_NZ2023_b2['tensors'].apply(is_empty_dict)]
data_NZ2023_c_final3 = missing_img_NZ2023_c2[~missing_img_NZ2023_c2['tensors'].apply(is_empty_dict)]
data_NZ2023_d_final3 = missing_img_NZ2023_d2[~missing_img_NZ2023_d2['tensors'].apply(is_empty_dict)]
# Reset the index
data_NZ2023_a_final3 = data_NZ2023_a_final3.reset_index(drop=True)
data_NZ2023_b_final3 = data_NZ2023_b_final3.reset_index(drop=True)
data_NZ2023_c_final3 = data_NZ2023_c_final3.reset_index(drop=True)
data_NZ2023_d_final3 = data_NZ2023_d_final3.reset_index(drop=True)



# In[167]:


missing_img_NZ2024_a2_unique = missing_img_NZ2024_a2.drop_duplicates(subset=['Spectral_folder_A'])
missing_img_NZ2024_b2_unique = missing_img_NZ2024_b2.drop_duplicates(subset=['Spectral_folder_B'])
missing_img_NZ2024_c2_unique = missing_img_NZ2024_c2.drop_duplicates(subset=['Spectral_folder_C'])
missing_img_NZ2024_d2_unique = missing_img_NZ2024_d2.drop_duplicates(subset=['Spectral_folder_D'])

missing_img_UK2024_a2_unique = missing_img_UK2024_a2.drop_duplicates(subset=['Spectral_folder_A'])
missing_img_UK2024_b2_unique = missing_img_UK2024_b2.drop_duplicates(subset=['Spectral_folder_B'])
missing_img_UK2024_c2_unique = missing_img_UK2024_c2.drop_duplicates(subset=['Spectral_folder_C'])
missing_img_UK2024_d2_unique = missing_img_UK2024_d2.drop_duplicates(subset=['Spectral_folder_D'])


# In[168]:


missing_img_NZ2024_a2['tensors'] = extractor_spectral2(missing_img_NZ2024_a2_unique, folder_link, NZ2024_loc_link, 'A')
missing_img_NZ2024_b2['tensors'] = extractor_spectral2(missing_img_NZ2024_b2_unique, folder_link, NZ2024_loc_link, 'B')
missing_img_NZ2024_c2['tensors'] = extractor_spectral2(missing_img_NZ2024_c2_unique, folder_link, NZ2024_loc_link, 'C') 
missing_img_NZ2024_d2['tensors'] = extractor_spectral2(missing_img_NZ2024_d2_unique, folder_link, NZ2024_loc_link, 'D')
# data_NZ2024_a_na['sorted_tensor'].head(5)



missing_img_UK2024_a2['tensors'] = extractor_spectral2(missing_img_UK2024_a2_unique, folder_link, UK2024_loc_link, 'A')
missing_img_UK2024_b2['tensors'] = extractor_spectral2(missing_img_UK2024_b2_unique, folder_link, UK2024_loc_link, 'B')
missing_img_UK2024_c2['tensors'] = extractor_spectral2(missing_img_UK2024_c2_unique, folder_link, UK2024_loc_link, 'C') 
missing_img_UK2024_d2['tensors'] = extractor_spectral2(missing_img_UK2024_d2_unique, folder_link, UK2024_loc_link, 'D')


# In[179]:


# Get df of missing images
# test_missing = test_data[test_data['tensors'].apply(is_empty_dict)]


missing_img_NZ2024_a3 = missing_img_NZ2024_a2[missing_img_NZ2024_a2['tensors'].apply(is_empty_dict)]
missing_img_NZ2024_b3 = missing_img_NZ2024_b2[missing_img_NZ2024_b2['tensors'].apply(is_empty_dict)]
missing_img_NZ2024_c3 = missing_img_NZ2024_c2[missing_img_NZ2024_c2['tensors'].apply(is_empty_dict)]
missing_img_NZ2024_d3 = missing_img_NZ2024_d2[missing_img_NZ2024_d2['tensors'].apply(is_empty_dict)]
missing_img_UK2024_a3 = missing_img_UK2024_a2[missing_img_UK2024_a2['tensors'].apply(is_empty_dict)]
missing_img_UK2024_b3 = missing_img_UK2024_b2[missing_img_UK2024_b2['tensors'].apply(is_empty_dict)]
missing_img_UK2024_c3 = missing_img_UK2024_c2[missing_img_UK2024_c2['tensors'].apply(is_empty_dict)]   
missing_img_UK2024_d3 = missing_img_UK2024_d2[missing_img_UK2024_d2['tensors'].apply(is_empty_dict)]


# In[180]:


len_nz2023_a = len(missing_img_NZ2023_a3)
len_nz2024_a = len(missing_img_NZ2024_a3)
len_uk2024_a = len(missing_img_UK2024_a3)
len_nz2023_b = len(missing_img_NZ2023_b3)
len_nz2024_b = len(missing_img_NZ2024_b3)
len_uk2024_b = len(missing_img_UK2024_b3)
len_nz2023_c = len(missing_img_NZ2023_c3)
len_nz2024_c = len(missing_img_NZ2024_c3)
len_uk2024_c = len(missing_img_UK2024_c3)
len_nz2023_d = len(missing_img_NZ2023_d3)
len_nz2024_d = len(missing_img_NZ2024_d3)
len_uk2024_d = len(missing_img_UK2024_d3)

pickle_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Pickle_files/Tensors/'


data = {
    'List Name': [
        'missing_img_unique_nz2023_a2', 'missing_img_unique_nz2024_a2', 'missing_img_unique_uk2024_a2',
        'missing_img_unique_nz2023_b2', 'missing_img_unique_nz2024_b2', 'missing_img_unique_uk2024_b2',
        'missing_img_unique_nz2023_c2', 'missing_img_unique_nz2024_c2', 'missing_img_unique_uk2024_c2',
        'missing_img_unique_nz2023_d2', 'missing_img_unique_nz2024_d2', 'missing_img_unique_uk2024_d2'
    ],
    'List Length': [
        len_nz2023_a, len_nz2024_a, len_uk2024_a,
        len_nz2023_b, len_nz2024_b, len_uk2024_b,
        len_nz2023_c, len_nz2024_c, len_uk2024_c,
        len_nz2023_d, len_nz2024_d, len_uk2024_d
    ]
}


df = pd.DataFrame(data)
df.to_csv(f'{pickle_path}summary3.csv', index=False)

# Save the missing lists to pickle files

save_if_not_empty('missing_img_NZ2024_a3', missing_img_NZ2024_a3)
save_if_not_empty('missing_img_NZ2024_b3', missing_img_NZ2024_b3)
save_if_not_empty('missing_img_NZ2024_c3', missing_img_NZ2024_c3)
save_if_not_empty('missing_img_NZ2024_d3', missing_img_NZ2024_d3)
save_if_not_empty('missing_img_UK2024_a3', missing_img_UK2024_a3)
save_if_not_empty('missing_img_UK2024_b3', missing_img_UK2024_b3)
save_if_not_empty('missing_img_UK2024_c3', missing_img_UK2024_c3)
save_if_not_empty('missing_img_UK2024_d3', missing_img_UK2024_d3)

print("The missing images have been saved to pickle files.")


# In[181]:


# # removing rows which miscount the number of apples in the image

# data_final_test = test_data[~test_data['tensors'].apply(is_empty_dict)]
# data_final_test = data_final_test.reset_index(drop=True)

data_NZ2024_a_final3 = missing_img_NZ2024_a3[~missing_img_NZ2024_a3['tensors'].apply(is_empty_dict)]
data_NZ2024_b_final3 = missing_img_NZ2024_b3[~missing_img_NZ2024_b3['tensors'].apply(is_empty_dict)]
data_NZ2024_c_final3 = missing_img_NZ2024_c3[~missing_img_NZ2024_c3['tensors'].apply(is_empty_dict)]
data_NZ2024_d_final3 = missing_img_NZ2024_d3[~missing_img_NZ2024_d3['tensors'].apply(is_empty_dict)]
# Reset the index
data_NZ2024_a_final3 = data_NZ2024_a_final3.reset_index(drop=True)
data_NZ2024_b_final3 = data_NZ2024_b_final3.reset_index(drop=True)
data_NZ2024_c_final3 = data_NZ2024_c_final3.reset_index(drop=True)
data_NZ2024_d_final3 = data_NZ2024_d_final3.reset_index(drop=True)


data_UK2024_a_final3 = missing_img_UK2024_a3[~missing_img_UK2024_a3['tensors'].apply(is_empty_dict)]
data_UK2024_b_final3 = missing_img_UK2024_b3[~missing_img_UK2024_b3['tensors'].apply(is_empty_dict)]
data_UK2024_c_final3 = missing_img_UK2024_c3[~missing_img_UK2024_c3['tensors'].apply(is_empty_dict)]
data_UK2024_d_final3 = missing_img_UK2024_d3[~missing_img_UK2024_d3['tensors'].apply(is_empty_dict)]
# Reset the index
data_UK2024_a_final3 = data_UK2024_a_final3.reset_index(drop=True)
data_UK2024_b_final3 = data_UK2024_b_final3.reset_index(drop=True)
data_UK2024_c_final3 = data_UK2024_c_final3.reset_index(drop=True)
data_UK2024_d_final3 = data_UK2024_d_final3.reset_index(drop=True)


# In[182]:


# Save data_final to pickle
pickle_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Pickle_files/Tensors/'

# data_final_test.to_pickle(f'{pickle_path}data_final_test.pkl')

data_NZ2023_a_final3.to_pickle(f'{pickle_path}data_NZ2023_a_final3.pkl')
data_NZ2023_b_final3.to_pickle(f'{pickle_path}data_NZ2023_b_final3.pkl')
data_NZ2023_c_final3.to_pickle(f'{pickle_path}data_NZ2023_c_final3.pkl')
data_NZ2023_d_final3.to_pickle(f'{pickle_path}data_NZ2023_d_final3.pkl')
data_NZ2024_a_final3.to_pickle(f'{pickle_path}data_NZ2024_a_final3.pkl')
data_NZ2024_b_final3.to_pickle(f'{pickle_path}data_NZ2024_b_final3.pkl')
data_NZ2024_c_final3.to_pickle(f'{pickle_path}data_NZ2024_c_final3.pkl')
data_NZ2024_d_final3.to_pickle(f'{pickle_path}data_NZ2024_d_final3.pkl')
data_UK2024_a_final3.to_pickle(f'{pickle_path}data_UK2024_a_final3.pkl')
data_UK2024_b_final3.to_pickle(f'{pickle_path}data_UK2024_b_final3.pkl')
data_UK2024_c_final3.to_pickle(f'{pickle_path}data_UK2024_c_final3.pkl')
data_UK2024_d_final3.to_pickle(f'{pickle_path}data_UK2024_d_final3.pkl')


# In[186]:


still_miss_NZ2023_a = len(missing_img_NZ2023_a2) - len_nz2023_a
still_miss_NZ2023_b = len(missing_img_NZ2023_b2) - len_nz2023_b
still_miss_NZ2023_c = len(missing_img_NZ2023_c2) - len_nz2023_c
still_miss_NZ2023_d = len(missing_img_NZ2023_d2) - len_nz2023_d
still_miss_NZ2024_a = len(missing_img_NZ2024_a2) - len_nz2024_a
still_miss_NZ2024_b = len(missing_img_NZ2024_b2) - len_nz2024_b
still_miss_NZ2024_c = len(missing_img_NZ2024_c2) - len_nz2024_c
still_miss_NZ2024_d = len(missing_img_NZ2024_d2) - len_nz2024_d
still_miss_UK2024_a = len(missing_img_UK2024_a2) - len_uk2024_a
still_miss_UK2024_b = len(missing_img_UK2024_b2) - len_uk2024_b
still_miss_UK2024_c = len(missing_img_UK2024_c2) - len_uk2024_c
still_miss_UK2024_d = len(missing_img_UK2024_d2) - len_uk2024_d

print(f'original: {len(missing_img_NZ2023_a2)}, remaining:{still_miss_NZ2023_a}')
print(f'original: {len(missing_img_NZ2023_b2)}, remaining:{still_miss_NZ2023_b}')
print(f'original: {len(missing_img_NZ2023_c2)}, remaining:{still_miss_NZ2023_c}')
print(f'original: {len(missing_img_NZ2023_d2)}, remaining:{still_miss_NZ2023_d}')
print(f'original: {len(missing_img_NZ2024_a2)}, remaining:{still_miss_NZ2024_a}')
print(f'original: {len(missing_img_NZ2024_b2)}, remaining:{still_miss_NZ2024_b}')
print(f'original: {len(missing_img_NZ2024_c2)}, remaining:{still_miss_NZ2024_c}')
print(f'original: {len(missing_img_NZ2024_d2)}, remaining:{still_miss_NZ2024_d}')
print(f'original: {len(missing_img_UK2024_a2)}, remaining:{still_miss_UK2024_a}')
print(f'original: {len(missing_img_UK2024_b2)}, remaining:{still_miss_UK2024_b}')
print(f'original: {len(missing_img_UK2024_c2)}, remaining:{still_miss_UK2024_c}')
print(f'original: {len(missing_img_UK2024_d2)}, remaining:{still_miss_UK2024_d}')









# In[188]:


print(len(missing_img_NZ2023_d2) )

