import argparse
import time
from pathlib import Path

import clip

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import xyxy2xywh, \
    strip_optimizer, set_logging, increment_path, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
from utils.roboflow import predict_image

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_clip_detections as gdet

from utils.yolov5 import Yolov5Engine




classes = []

names = []


def update_tracks(tracker, frame_count, save_txt, txt_path, save_img, view_img, im0, gn):
    if len(tracker.tracks):
        print("[Tracks]", len(tracker.tracks))

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        xyxy = track.to_tlbr()
        class_num = track.class_num
        bbox = xyxy
        class_name = names[int(class_num)] if opt.detection_engine == "yolov5" else class_num
        if opt.info:
            print("Tracker ID: {}, Class: {}, BBox Coords (xmin, ymin, xmax, ymax): {}".format(
                str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        if save_txt:  # Write to file
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                              ) / gn).view(-1).tolist()  # normalized xywh

            with open(txt_path + '.txt', 'a') as f:
                f.write('frame: {}; track: {}; class: {}; bbox: {};\n'.format(frame_count, track.track_id, class_num,
                                                                              *xywh))

        if save_img or view_img:  # Add bbox to image
            label = f'{class_name} #{track.track_id}'
            plot_one_box(xyxy, im0, label=label,
                         color=get_color_for(label), line_thickness=opt.thickness)

def get_color_for(class_num):
    colors = [
        "#4892EA",
        "#00EEC3",
        "#FE4EF0",
        "#F4004E",
        "#FA7200",
        "#EEEE17",
        "#90FF00",
        "#78C1D2",
        "#8C29FF"
    ]

    num = hash(class_num) # may actually be a number or a string
    hex = colors[num%len(colors)]

    # adapted from https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
    rgb = tuple(int(hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    return rgb

def detect(save_img=False):

    t0 = time_synchronized()

    nms_max_overlap = opt.nms_max_overlap
    max_cosine_distance = opt.max_cosine_distance
    nn_budget = opt.nn_budget

    # initialize deep sort
    model_filename = "ViT-B/32"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    half = device != "cpu"
    model, transform = clip.load(model_filename, device=device)
    encoder = gdet.create_box_encoder(model, transform, batch_size=1, device=device)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    
    # load yolov5 model here
    if opt.detection_engine == "yolov5":
        print("loading yolov5 model")
        yolov5_engine = Yolov5Engine(opt.weights, device, opt.classes, opt.confidence, opt.overlap, opt.agnostic_nms, opt.augment, half)
        global names
        names = yolov5_engine.get_names()
    elif opt.detection_engine == "yolov4":
        # Test TF Imports
        from yolov4 import tf_nms, process_detections
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        from absl import app, flags, logging
        from absl.flags import FLAGS
        import core.utils as utils
        from core.yolov4 import filter_boxes
        from tensorflow.python.saved_model import tag_constants
        from core.config import cfg
        from PIL import Image
        import matplotlib.pyplot as plt
        from tensorflow.compat.v1 import ConfigProto
        from tensorflow.compat.v1 import InteractiveSession
        # load configuration for object detector
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

        # load tflite model if flag is set
        if opt.framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=opt.weights[0])
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
        # otherwise load standard tensorflow saved model
        else:
            saved_model_loaded = tf.saved_model.load(opt.weights[0], tags=[tag_constants.SERVING])
            infer = saved_model_loaded.signatures['serving_default']

    # initialize tracker
    tracker = Tracker(metric)

    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name,
                    exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                          exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    frame_count = 0
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    if opt.detection_engine == "yolov5":
        _ = yolov5_engine.infer(img.half() if half else img) if device.type != 'cpu' else None  # run once
    elif opt.detection_engine == "yolov4":
        print("yolov4 test inference")
        '''image_data = np.random.rand(416, 416, 3)
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        pred = yolov4_engine.infer_img(image_data)
        print("yolov4 test inference done")'''

    for path, img, im0s, vid_cap in dataset:

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Roboflow Inference
        t1 = time_synchronized()
        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

        # choose between prediction engines (yolov5 and roboflow)
        if opt.detection_engine == "roboflow":
            pred, classes = predict_image(im0, opt.api_key, opt.url, opt.confidence, opt.overlap, frame_count)
            pred = [torch.tensor(pred)]
        elif opt.detection_engine == "yolov5":
            pred = yolov5_engine.infer(img)
        elif opt.detection_engine == "yolov4":
            image_data = cv2.resize(im0, (416, 416))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)

            # run detections on tflite if flag is set
            if opt.framework == 'tflite':
                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                # run detections using yolov3 if flag is set
                if opt.model == 'yolov3' and opt.tiny == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
            else:
                batch_data = tf.constant(image_data)
                pred_bbox = infer(batch_data)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]

                    pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf_nms(boxes, pred_conf, opt.overlap, opt.confidence)

            pred = [[]]



        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            #moved up to roboflow inference
            """if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(
                ), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)"""

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + \
                ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt

            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            print("\n[Detections]")
            if opt.detection_engine == "yolov4":
                detections = process_detections(boxes, scores, classes, valid_detections, im0, opt, encoder, s)
                # Call the tracker
                tracker.predict()
                tracker.update(detections)

                # update tracks
                update_tracks(tracker, frame_count, save_txt, txt_path, save_img, view_img, im0, gn)

            else:
                if len(det):


                    if opt.detection_engine == "roboflow":
                        # Print results
                        clss = np.array(classes)
                        for c in np.unique(clss):
                            n = (clss == c).sum()  # detections per class
                            s += f'{n} {c}, '  # add to string

                        trans_bboxes = det[:, :4].clone()
                        bboxes = trans_bboxes[:, :4].cpu()
                        confs = det[:, 4]

                    else:
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f'{n} {names[int(c)]}s, '  # add to string

                        # Transform bboxes from tlbr to tlwh
                        trans_bboxes = det[:, :4].clone()
                        trans_bboxes[:, 2:] -= trans_bboxes[:, :2]
                        bboxes = trans_bboxes[:, :4]
                        confs = det[:, 4]
                        class_nums = det[:, -1]
                        classes = class_nums

                        print(s)



                    # encode yolo detections and feed to tracker
                    features = encoder(im0, bboxes)
                    detections = [Detection(bbox.cpu(), conf, class_num, feature) for bbox, conf, class_num, feature in zip(
                        bboxes, confs, classes, features)]

                    # run non-maxima supression
                    boxs = np.array([d.tlwh for d in detections])
                    scores = np.array([d.confidence for d in detections])
                    class_nums = np.array([d.class_num for d in detections])
                    indices = preprocessing.non_max_suppression(
                        boxs, class_nums, nms_max_overlap, scores)
                    detections = [detections[i] for i in indices]


                    # Call the tracker
                    tracker.predict()
                    tracker.update(detections)

                    # update tracks
                    update_tracks(tracker, frame_count, save_txt, txt_path, save_img, view_img, im0, gn)

            # Print time (inference + NMS)
            print(f'Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

            frame_count = frame_count+1

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='yolov5s.pt', help='model.pt path(s)')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='data/images', help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--confidence', type=float,
                        default=0.40, help='object confidence threshold')
    parser.add_argument('--overlap', type=float,
                        default=0.30, help='IOU threshold for NMS')
    parser.add_argument('--thickness', type=int,
                        default=3, help='Thickness of the bounding box strokes')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--nms_max_overlap', type=float, default=1.0,
                        help='Non-maxima suppression threshold: Maximum detection overlap.')
    parser.add_argument('--max_cosine_distance', type=float, default=0.4,
                        help='Gating threshold for cosine distance metric (object appearance).')
    parser.add_argument('--nn_budget', type=int, default=None,
                        help='Maximum size of the appearance descriptors allery. If None, no budget is enforced.')
    parser.add_argument('--api_key', default=None,
                        help='Roboflow API Key.')
    parser.add_argument('--url', default=None,
                        help='Roboflow Model URL.')
    parser.add_argument('--info', action='store_true',
                        help='Print debugging info.')
    parser.add_argument("--detection-engine", default="roboflow", help="Which engine you want to use for object detection (yolov5, yolov4, roboflow).")
    parser.add_argument('--framework', default='tf', help='(tf, tflite')
    parser.add_argument('--size', default=416, help='resize images to')
    parser.add_argument('--tiny', default=False, help='yolo or yolo-tiny')
    parser.add_argument('--model', default='yolov4', help='yolov3 or yolov4')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
