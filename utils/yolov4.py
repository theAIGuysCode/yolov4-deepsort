from models.experimental import attempt_load
from utils.general import non_max_suppression
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf

class Yolov4Engine:
    def __init__(self, weights, device, classes, conf_thres, iou_thres, agnostic_nms, augment, half, framework, model):
        self.classes = classes
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.augment = augment
        self.agnostic_nms = agnostic_nms
        self.framework = framework
        self.model = model

        # load tflite model if flag is set
        if framework == 'tflite':
            self.interpreter = tf.lite.Interpreter(model_path=weights)
            self.interpreter.allocate_tensors()
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            print(input_details)
            print(output_details)
        # otherwise load standard tensorflow saved model
        else:
            print("[YOLOv4Engine __init__] init model; weights: {}".format(weights[0]))
            saved_model_loaded = tf.saved_model.load(weights[0], tags=[tag_constants.SERVING])
            self.infer = saved_model_loaded.signatures['serving_default']

    def infer(self, img):
        # run detections on tflite if flag is set
        print("yolov4 infer func")
        img = img.numpy()
        if self.framework == 'tflite':
            self.interpreter.set_tensor(input_details[0]['index'], image_data)
            self.interpreter.invoke()
            pred = [self.interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if self.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            print("[YOLOv4Engine infer] img: {}".format(img))
            #batch_data = tf.constant(img[0])
            print("[YOLOv4Engine infer] batch_data: {}".format(batch_data))
            #pred_bbox = self.infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        return nms(boxes, pred_conf)

    def nms(self, boxes, pred_conf):
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        return boxes, scores, classes, valid_detections