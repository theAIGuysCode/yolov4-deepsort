# Roboflow Object Tracking Example

Object tracking using Roboflow Inference API and zeroshot CLIP Deep SORT. [More on CLIP](https://blog.roboflow.com/how-to-use-openai-clip/)

![Tracking Fish](example/video/fish.mp4?raw=True)

Example object tracking courtesy of the [Roboflow Universe public Aquarium model and dataset](https://universe.roboflow.com/brad-dwyer/aquarium-combined). You can adapt this to your own dataset on Roboflow or any pre-trained model from [Roboflow Universe](https://universe.roboflow.com).

# Overview

Object tracking involves following individual objects of interest across frames. It
combines the output of an [object detection](https://blog.roboflow.com/object-detection) model
with a secondary algorithm to determine which detections are identifying "the same"
object over time.

Previously, this required training a special classification model to differentiate
the instances of each different class. In this repository, we have used
[OpenAI's CLIP zero-shot image classifier](https://blog.roboflow.com/clip-model-eli5-beginner-guide/)
to create a universal object tracking repository. All you need is a trained object
detection model and CLIP handles the instance identification for the object tracking
algorithm.

# Getting Started

## Training your model

Upload, annotate, and train your model on Roboflow with [Roboflow Train](https://docs.roboflow.com/train). Your model will be hosted on an inference URL.

## Performing Object Tracking

Clone repositories

```
git clone https://github.com/roboflow-ai/zero-shot-object-tracking
cd zero-shot-object-tracking
git clone https://github.com/openai/CLIP.git CLIP-repo
cp -r ./CLIP-repo/clip ./clip # Unix based
robocopy CLIP-repo/clip clip\ # Windows
```

Install requirements (python 3.7+)

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Install requirements (anaconda python 3.8)
```
conda install pytorch torchvision torchaudio -c pytorch
conda install ftfy regex
pip install opencv pycocotools tensorflow
```

Run

```bash
python clip_object_tracker.py --source data/video/cards.mp4 --url https://detect.roboflow.com/playing-cards-ow27d/1 --api_key ROBOFLOW_API_KEY
```

(by default, output will be in runs/detect/exp[num])

<figure class="video_container">
  <video controls="true" allowfullscreen="true" poster="path/to/poster_image.png">
    <source src="data/demo/cards.mp4" type="video/mp4">
  </video>
</figure>

Help

```bash
python clip_object_tracker.py -h
```

## Acknowledgements

Huge thanks to:

- [yolov4-deepsort by theAIGuysCode](https://github.com/theAIGuysCode/yolov4-deepsort)
- [yolov5 by ultralytics](https://github.com/ultralytics/yolov5)
- [Deep SORT Repository by nwojke](https://github.com/nwojke/deep_sort)
- [OpenAI for being awesome](https://openai.com/blog/clip/)
