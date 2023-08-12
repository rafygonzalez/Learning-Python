import os
import glob
import uuid

folder_path = ""  # Replace with the actual folder path

# Find all PNG files in the folder
png_files = glob.glob(os.path.join(folder_path, "*.png"))

# Enumerate and rename the PNG files
for _, old_name in enumerate(png_files):
    uuidstr = str(uuid.uuid4())
    filename = uuidstr

    new_name = os.path.join(folder_path, f"{uuidstr}.png")  # Change the naming pattern if desired
    os.rename(old_name, new_name)

print("File renaming completed!")
