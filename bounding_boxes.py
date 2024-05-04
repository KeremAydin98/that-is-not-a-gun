from transformers import pipeline
from PIL import Image, ImageDraw
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torch

class OwlVit:

    def __init__(self, ):

        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    def detect(self, image, text_prompt):

        inputs = self.processor(text=text_prompt, images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = self.processor.post_process_object_detection(outputs=outputs, threshold=0.1, target_sizes=target_sizes)

        return results

    def show_boxes_and_labels(self, image, results, text_prompt):

        draw = ImageDraw.Draw(image)   

        bboxes = []

        for i in range(len(results)):

            highest_score = torch.argmax(results[i]['scores']).item()

            box = results[i]['boxes'][highest_score]

            x1, y1, x2, y2 = box[0].item(), box[1].item(), box[2].item(), box[3].item()

            bboxes.append([x1, y1, x2, y2])
            shape = [(x1, y1), (x2, y2)]

            draw.rectangle(shape, outline ="green") 
            draw.text((x1, y1-10), text_prompt[i], align ="left")  

    
        image = image.save("/mnt/keremaydin/that-is-not-a-gun/gun_image_with_bb.jpg")

        return bboxes