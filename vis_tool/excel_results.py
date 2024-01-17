import glob
import re
import csv

def find_accuracy_lines(file_path):
    pattern = r"\* Acc@1 ([\d\.]+) Acc@5 ([\d\.]+) loss ([\d\.]+)"
    acc1_values = [file_path]  # Include the file path in the output
    epoch = 1

    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match and epoch % 10 == 0:  # Only append if epoch is a multiple of 5
                acc1 = float(match.group(1))
                acc1_values.append(acc1)
            epoch += 1

    return acc1_values

# Specify your target folder path
target_folder_path = '/raid/home_yedek/utku/ViTFreeze/ViT/finetune'

# Step 1: Find all cli_logs.txt files in the target folder
file_paths = glob.glob(f'{target_folder_path}/**/cli_logs.txt', recursive=True)

# Step 2 and 3: Read each file, extract the accuracy lines, and write to CSV
with open('accuracy_results.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    # Write header row
    header = ['File Path']
    header.extend([f'Epoch {i}' for i in range(10, 110, 10)])  # Assuming a maximum of 100 epochs
    csvwriter.writerow(header)

    for path in file_paths:
        acc1_values = find_accuracy_lines(path)
        csvwriter.writerow(acc1_values)
