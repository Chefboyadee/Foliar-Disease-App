import os
import rembg
from PIL import Image
import io

def remove_background(input_path, output_path):
    with open(input_path, "rb") as input_file:
        input_data = input_file.read()

        
        output_data = rembg.remove(input_data)

       
        output_image = Image.open(io.BytesIO(output_data))

 
        output_image.save(output_path)

def process_images(input_folder, output_folder):
   
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

   
    for filename in os.listdir(input_folder):
        if filename.endswith(".JPG") or filename.endswith(".jpeg"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_no_bg.png")

            remove_background(input_path, output_path)
            print(f"Processed: {filename}")

if __name__ == "__main__":
   
    input_folder_path = "Images/Test"
    output_folder_path = "Images/New folder"

    process_images(input_folder_path, output_folder_path)
