# Siamese YOLO, from Jayce

from ultralytics.engine.model import Model
from ultralytics.models import siayolo
from ultralytics.nn.tasks import SiameseSegModel

class Siamese(Model):
    """Siamese YOLO segmentation model"""

    @property
    def task_map(self):
        return {
            'segment': {
                'model': SiameseSegModel,
                'trainer': siayolo.segment.SiameseSegmentationTrainer,
                'validator': siayolo.segment.SegmentationValidator,
                'predictor': siayolo.segment.SegmentationPredictor,
            }
        }