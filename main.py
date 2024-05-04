from bounding_boxes import *
from generate_masks import * 
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
detector = OwlVit()
mask_generator = SAM_mask_generator()

# Load the image
image = Image.open('/mnt/keremaydin/that-is-not-a-gun/gun_image.jpg')

# Text prompt
text_prompt = ['a photo of a gun']

# Inference
results = detector.detect(image, text_prompt)

# View the result
bboxes = detector.show_boxes_and_labels(image, results, text_prompt)

# Generate masks
masks = mask_generator.generate_mask(image, results, bboxes[0], text_prompt)

# Obtain only a single mask
image_mask = torch.zeros(masks[0].shape)

for mask in masks:
    image_mask = torch.add(image_mask, mask)

# Initialize Stable Diffusion 
sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "./models/stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16,
        low_cpu_mem_usage=False if torch.cuda.is_available() else True,
    ).to(device)

# Set the value of seed manually for reproducibility of the results
seed = 66733
generator = torch.Generator(device).manual_seed(seed)

prompt = "a banana in the shape of gun"

output = sd_pipe(
  image=image,
  mask_image=image_mask,
  prompt=prompt,
  generator=generator,
  num_inference_steps=3,
)

generated_image = output.images[0]

generated_image = generated_image.save("/mnt/keremaydin/that-is-not-a-gun/gun_replaced_with_banana.jpg")

print(masks)
