from diffusers import StableDiffusionInpaintPipeline
import torch
import numpy as np

class Diffusion:

    def __init__(self, device):


        self.device = device
        # Initialize Stable Diffusion 
        self.sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                revision="fp16",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=False
            ).to(self.device)

    def replace_mask_with_image(self, image_tensor, image_mask, prompt):

        # Set the value of seed manually for reproducibility of the results
        generator = torch.Generator(self.device).manual_seed(61)

        #image_tensor = image_tensor.to(self.device)
        #image_mask = image_mask.to(self.device)

        generated_image = self.sd_pipe(
            image=image_tensor,
            mask_image=image_mask,
            prompt=prompt,
            generator=generator,
            num_inference_steps=100,
            guidance_scale=25
            ).images[0]

        #generated_image.save("/mnt/keremaydin/that-is-not-a-gun/generated_image.jpg")

        return generated_image