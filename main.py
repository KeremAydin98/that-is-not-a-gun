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