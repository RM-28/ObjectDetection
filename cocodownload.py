import fiftyone as fo
import fiftyone.zoo as foz
from pylabel import importer
import os

# Train Annotations: "C:\\Users\\Kenny Liang\\fiftyone\\coco-2017\\train\\labels.json"
# Train Images: "C:\\Users\\Kenny Liang\\fiftyone\\coco-2017\\train\data"
# Validation Annotations: "C:\\Users\Kenny Liang\\fiftyone\\coco-2017\\validation\\labels.json"
# Validation Images: "C:\\Users\\Kenny Liang\\fiftyone\\coco-2017\\validation\\data"

# Save directory of train: "C:\\Users\\Kenny Liang\\ObjectDetection\\datasets\\finalDataset\\train\\labels"
# Save directory of val: "C:\\Users\\Kenny Liang\\ObjectDetection\\datasets\\finalDataset\\valid\\labels"

#----------------------------------------------------------------#
ann_path = "C:\\Users\Kenny Liang\\fiftyone\\coco-2017\\validation\\labels.json"
coco_images_dir = "C:\\Users\\Kenny Liang\\fiftyone\\coco-2017\\validation\\data"
save_task_dir = "C:\\Users\\Kenny Liang\\ObjectDetection\\datasets\\tempFolder"


# dataset = importer.ImportCoco(path=ann_path, path_to_images=coco_images_dir)
# print(f"Number of images: {dataset.analyze.num_images}")
# print(f"Number of classes: {dataset.analyze.num_classes}")
# print(f"Classes:{dataset.analyze.classes}")
# print(f"Class counts:\n{dataset.analyze.class_counts}")
# print(f"Path to annotations:\n{dataset.path_to_annotations}")
# dataset.export.ExportToYoloV5(save_task_dir)

#----------------------------------------------------------------#

# dataset = foz.load_zoo_dataset(
#     "coco-2017",
#     split="test",
#     label_types=["detections"],
#     classes = ["handbag", "tie", "bottle", "cup", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock"],
#     max_samples=1200,
#     shuffle=True,
# )
# 6000 1700 800
# 4000 1100 500
# 3000 850 400
#----------------------------------------------------------------#






def filter_rows_in_folder(folder_path, specified_numbers=None):
    if specified_numbers is None:
        specified_numbers = [1]  # Default to [1] if no numbers are provided

    # Convert specified numbers to strings for comparison
    # I think we can add 1 here VVVV to match it with the yolov5 format
    specified_numbers = [(str(num)+ " ") for num in specified_numbers]

    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):  # Process only .txt files
            file_path = os.path.join(folder_path, filename)  # Get full file path
            with open(file_path, 'r') as infile:
                lines = infile.readlines()
            filtered_lines = [line for line in lines if any(line.lstrip().startswith(num) for num in specified_numbers)]
            with open(file_path, 'w') as outfile:
                outfile.writelines(filtered_lines)
                


def replace_first_column_in_folder(folder_path, specified_number, replacement_number):
    # Convert specified number to a string for comparison
    specified_number = str(specified_number)
    replacement_number = str(replacement_number)

    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):  # Process only .txt files
            file_path = os.path.join(folder_path, filename)  # Get full file path

            with open(file_path, 'r') as infile:
                lines = infile.readlines()
                

            updated_lines = []
            for line in lines:
                stripped_line = line.strip()
                parts = stripped_line.split()
                if parts and parts[0] == specified_number:
                    # Replace the first column with the replacement number
                    parts[0] = replacement_number
                    new_line = ' '.join(parts)  # Rejoin the line with the replaced first column
                    updated_lines.append(new_line + '\n')
                else:
                    updated_lines.append(line)  # Keep the line unchanged if the first column doesn't match

            # Write the updated lines back to the same file
            with open(file_path, 'w') as outfile:
                outfile.writelines(updated_lines)


folder_path = "C:\\Users\\Kenny Liang\\ObjectDetection\\datasets\\doordataset\\valid\\labels"
input_folder = "C:\\Users\\Kenny Liang\\ObjectDetection\\datasets\\tempFolder"
specified_number = 0
replacement_number = 92   

if __name__ == '__main__':
    # filter_rows_in_folder(input_folder, [1, 31, 32, 44, 47, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85])
    replace_first_column_in_folder(folder_path, specified_number, replacement_number)


# [1, 31, 32, 44, 47, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85]
