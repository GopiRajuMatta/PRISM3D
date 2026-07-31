import os
import numpy as np
from PIL import Image

def average_images_in_folder(folder_path, output_name="average.png"):
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if not image_files:
        print("No images found.")
        return

    # Load the first image to get size and mode
    first_image = Image.open(os.path.join(folder_path, image_files[0]))
    img_array = np.array(first_image).astype(np.float32)
    count = 1

    # Accumulate the rest
    for fname in image_files[1:]:
        img_path = os.path.join(folder_path, fname)
        img = Image.open(img_path)
        img_array += np.array(img).astype(np.float32)
        count += 1

    # Average
    avg_array = (img_array / count).astype(np.uint8)
    avg_image = Image.fromarray(avg_array)

    # Save result
    output_path = os.path.join(folder_path, output_name)
    avg_image.save(output_path)
    print(f"Averaged image saved to {output_path}")

# Example usage:
average_images_in_folder("/data/Gopi/Gopi/PRISM3D/results/e2nerf_exblurf_glomap_events/bench/renders/1/render_traj")

