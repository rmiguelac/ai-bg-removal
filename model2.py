from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize
import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
import io
from skimage import io
import gc
import os

#input_dir = "/home/rui/repositories/bg-removal/input"
#output_dir = "/home/rui/repositories/bg-removal/output"
input_dir = "/app/input"
output_dir = "/app/output"

def run():
  torch.cuda.empty_cache()
  gc.collect()

  model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4",trust_remote_code=True)

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

  def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
      # Convert numpy array to PIL Image
      im = Image.fromarray(im)
  
      # Resize the image if necessary
      max_size = 1024
      if max(im.size) > max_size:
          im.thumbnail((max_size, max_size))
  
      # Convert back to numpy array
      im = np.array(im)
  
      if len(im.shape) < 3:
          im = im[:, :, np.newaxis]
  
      im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
      im_tensor = F.interpolate(torch.unsqueeze(im_tensor,0), size=model_input_size, mode='bilinear')
      image = torch.divide(im_tensor,255.0)
      image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])
      return image
  
  def postprocess_image(result: torch.Tensor, im_size: list)-> np.ndarray:
      result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear') ,0)
      ma = torch.max(result)
      mi = torch.min(result)
      result = (result-mi)/(ma-mi)
      im_array = (result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8)
      im_array = np.squeeze(im_array)
      return im_array

  for filename in os.listdir(input_dir):
      if filename.endswith(".png"):  # Add or modify to include all desired file types
            image_path = os.path.join(input_dir, filename)
  
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)
  
            print("1")
            # prepare input
            orig_im = io.imread(image_path)
            orig_im_size = orig_im.shape[0:2]
            model_input_size = [1024, 1024]
            image = preprocess_image(orig_im, model_input_size).to(device)
  
            print("2")
            # inference 
            result=model(image)
  
            print("3")
            # post process
            result_image = postprocess_image(result[0][0], orig_im_size)
  
            print("4")
            # save result
            pil_im = Image.fromarray(result_image)
            no_bg_image = Image.new("RGBA", pil_im.size, (0,0,0,0))
            orig_image = Image.open(image_path)
            no_bg_image.paste(orig_image, mask=pil_im)
            no_bg_image.save(os.path.join(output_dir,filename))
  
