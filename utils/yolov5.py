from models.experimental import attempt_load
from general import non_max_suppression

class Yolov5Engine:
    def __init__(self, weights, device, half, classes, conf_thres, iou_thres, agnostic_nms):
        self.model = attempt_load(weights, map_location=device)
        self.model.half()
        self.classes = classes
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def infer(self, img, augment):
        pred = self.model(img, augment=augment)[0]
        pred = self.nms(pred)
        return pred

    def nms(self, pred):
        out = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        return out
