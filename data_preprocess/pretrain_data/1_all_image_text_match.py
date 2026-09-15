import os
import csv
def find_image_paths(root_dir):
    image_paths = []
    for folder_name in range(10, 20):
        folder_path = os.path.join(root_dir, f"p{folder_name:02d}")
        if os.path.isdir(folder_path):
            for root, _, files in os.walk(folder_path):
                image_paths.extend([os.path.join(root, file) for file in files if file.lower().endswith('.jpg')])
    return image_paths

def find_text_paths(root_dir):
    text_paths = []
    for folder_name in range(10, 20):
        folder_path = os.path.join(root_dir, f"p{folder_name:02d}")
        if os.path.isdir(folder_path):
            for root, _, files in os.walk(folder_path):
                text_paths.extend([os.path.join(root, file) for file in files if file.lower().endswith('.txt')])
    return text_paths
def match_paths(image_paths, text_paths):
    matched_paths = []
    path_dict = {}
    for text in text_paths:
        text_name = os.path.basename(text)
        key = os.path.splitext(text_name)[0]
        path_dict[key] = text
    for image in image_paths:
        folder_path = os.path.dirname(image)
        folder_name = os.path.split(folder_path)[-1]
        if folder_name in path_dict:
            matched_paths.append((image, path_dict[folder_name]))
    return matched_paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.join(BASE_DIR, "./MIMIC_data/mimic-cxr-jpg")
image_paths = find_image_paths(root_directory)
print(len(image_paths))
root_directory_text = os.path.join(BASE_DIR, "./MIMIC_data/mimic-cxr-txt")
text_paths = find_text_paths(root_directory_text)
print(len(text_paths))

matched_paths = match_paths(image_paths, text_paths)
print(len(matched_paths))
csv_file_path = os.path.join(BASE_DIR, "./result/1_all_matched.csv")
with open(csv_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["img_path", "txt_path"])
    writer.writerows(matched_paths)