from transformers import pipeline
from PIL import Image, ImageDraw
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torch

class OwlVit:

    def __init__(self):

        '''self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")'''

        self.detector = pipeline(
            model= "google/owlvit-base-patch32",
            task="zero-shot-object-detection",
        )

    def detect(self, image, text_prompt):

        '''inputs = self.processor(text=text_prompt, images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([(512, 512)])
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = self.processor.post_process_object_detection(outputs=outputs, threshold=0.05, target_sizes=target_sizes)'''

        output = self.detector(
                    image,
                    candidate_labels=[text_prompt])

        return output

    def show_boxes_and_labels(self, image, results, text_prompt):

        draw = ImageDraw.Draw(image)   

        bboxes = []
        points = []

        for i in range(len(results)):

            box = results[i]['box']

            x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']

            point = [(x1 + x2) // 2, (y1 + y2) // 2]

            points.append(point)

            # Define the radius of the circle (to make the point bigger)
            radius = 5

            # Define the color (red)
            point_color = (255, 0, 0)  # RGB value for red

            # Calculate the bounding box of the circle
            circle = (point[0] - radius, point[1] - radius,
                    point[0] + radius, point[1] + radius)

            bboxes.append([x1, y1, x2, y2])
            # Draw the filled circle
            #draw.ellipse(circle, fill=point_color)
            shape = [(x1, y1), (x2, y2)]

            #draw.rectangle(shape, outline ="green") 
            #draw.text((x1, y1-10), text_prompt, align ="left")  

    
        #image.save("/mnt/keremaydin/that-is-not-a-gun/bounding_box.jpg")

        return points, bboxes