from bounding_boxes import *
from generate_masks import * 
from generate_image import *
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")


class ReplaceAll:

    def __init__(self, device, will_replace, replace_with):

        self.device = device
        self.text_prompt = will_replace
        self.replace_with = replace_with
        self.load_models()

    def load_models(self):

        # Load models
        self.detector = OwlVit()
        self.mask_generator = SAM_mask_generator(device=self.device)
        self.diffusion = Diffusion(device=self.device)

    def replace(self, image):

        results = []
        prompt = ''

        # Inference
        for text_prompt in self.text_prompt:
            result = self.detector.detect(image, text_prompt)

            if len(result) != 0:
                results = result
                prompt = text_prompt
                break

        if len(results) == 0:
            return image
            
        # View the result
        points, bboxes = self.detector.show_boxes_and_labels(image, results, prompt)

        # Generate masks
        image_mask = self.mask_generator.generate_mask(image, results, points[0], bboxes[0], prompt)

        generated_image = self.diffusion.replace_mask_with_image(image, image_mask, self.replace_with)

        return generated_image