import os
import shutil

# Base dataset path
target_base_folder = "/data/Gopi/Gopi/PRISM3D/data/exblurf_glomap"

# Subfolder names to process
subfolders = ["bench", "camellia", "dragon", "jars", "jars2", "postbox", "stone_lantern", "sunflowers"]

def replace_all_with_first_image():
    for subfolder in subfolders:
        folder_path = os.path.join(target_base_folder, subfolder)

        # Find numbered subfolders (e.g., "0", "1", ...)
        numbered_folders = [f for f in os.listdir(folder_path) if f.isdigit()]
        numbered_folders = sorted(numbered_folders, key=lambda x: int(x))

        for num_folder in numbered_folders:
            num_folder_path = os.path.join(folder_path, num_folder,"images")

            if not os.path.exists(num_folder_path):
                print(f"Folder does not exist: {num_folder_path}")
                continue

            image_files = sorted([
                f for f in os.listdir(num_folder_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])


            if not image_files:
                print(f"No images found in {num_folder_path}")
                continue

            first_image_path = os.path.join(num_folder_path, image_files[0])
            with open(first_image_path, 'rb') as f:
                first_image_data = f.read()

            for image_file in image_files:
                image_path = os.path.join(num_folder_path, image_file)
                with open(image_path, 'wb') as f:
                    f.write(first_image_data)

            print(f"Replaced all images in {num_folder_path} with {image_files[0]}")

if __name__ == "__main__":
    replace_all_with_first_image()

