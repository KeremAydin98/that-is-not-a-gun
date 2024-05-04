from ultralytics import SAM
import numpy as np

class SAM_mask_generator:

    def __init__(self):

        self.model = SAM('mobile_sam.pt')

    def generate_mask(self, image, results, bbox, labels):

        labels = np.repeat(1, len(results))

        mask_result = self.model.predict(
            image,
            bboxes=bbox,
            labels=labels
        )

        masks = mask_result[0].masks.data

        return masks



