from ultralytics import SAM
from transformers import SamModel, SamProcessor
import torch
import torchvision.transforms as transforms
import numpy as np

class SAM_mask_generator:

    def __init__(self, device):

        #self.model = SAM('sam_b.pt')
        self.device = device
        
        self.model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device=self.device)
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    def generate_mask(self, image, results, point, bbox, labels):

        '''labels = np.repeat(1, len(results))

        mask_result = self.model(
            image,
            points=point,
            labels=labels
        )

        masks = mask_result[0].masks.data'''

        inputs = self.processor(image, input_points=[[point]], input_boxes=[[bbox]], return_tensors="pt").to(device=self.device)
        outputs = self.model(**inputs)
        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0]
        #scores = outputs.iou_scores

        # Obtain only a single mask
        image_mask = torch.zeros(masks[0].size())

        for mask in masks[0]:
            image_mask = torch.add(image_mask, mask)

        image_pil = transforms.ToPILImage()(image_mask.cpu().detach().clamp_(0, 1))
        #image_pil.save("/mnt/keremaydin/that-is-not-a-gun/mask_image.jpg")

        #self.show_mask(masks[0])

        return image_pil

    def show_mask(self, mask, random_color=False):

        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        
        mask_image.save("/mnt/keremaydin/that-is-not-a-gun/mask_image.jpg")



