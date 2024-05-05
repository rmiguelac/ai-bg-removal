import os
from PIL import Image
from transformers import pipeline

def remove_bg():
  # Define the input and output directories
  input_dir = "/home/rui/repositories/bg-removal/input"
  output_dir = "/home/rui/repositories/bg-removal/output"
  
  # Initialize the pipeline
  pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
  
  # Iterate over all files in the input directory
  for filename in os.listdir(input_dir):
      if filename.endswith(".jpg"):  # Add or modify to include all desired file types
          image_path = os.path.join(input_dir, filename)

          print("converting")
          # Convert the image to PNG
          img = Image.open(image_path)
          rgb_img = img.convert('RGB')
          png_image_path = os.path.splitext(image_path)[0] + ".png"
          rgb_img.save(png_image_path)
          print("converting done")

  for filename in os.listdir(input_dir):
      if filename.endswith(".png"):  # Add or modify to include all desired file types
            image_path = os.path.join(input_dir, filename)
            print(rgb_img.mode)
            print("Starting to remove bg")
            # Process the image and get the mask and image
            #pillow_mask = pipe(png_image_path, return_mask=True)
            pillow_image = pipe(image_path)
  
            # Save the mask and image to the output directory
            #mask_output_path = os.path.join(output_dir, f"mask_{filename}")
            image_output_path = os.path.join(output_dir, filename)
  
            #pillow_mask.save(mask_output_path)
            pillow_image.save(image_output_path)