## Deep Learning Project for Robotics Applications

#Project: Comparison of performance between Vanilla CNN( 3 Layers) with YoloV8.


Testing steps:
1. Clone the repository
  ```
git clone https://github.com/suryaveer01/DeepLearningforRobotics.git
  ```


2. Create a new Conda Env
```
conda create -n testenv python=3.7
```
3. Activate new Conda Env
```
conda activate testenv
```
4. Install requirements
```
pip install -r .\requirements.txt
```
5. Run test Script to test the fully trained model
```
python .\test.py 
```
6. Run train script to start training a new model(will not replace the existing model but will create a new file)
```
python .\train.py
```
7. Check performance of YOLOv8 Classifier trained on Imagenet
```
python .\yoloimageclassification.py
```

8. Check tensorboard
```
tensorboard --logdir=runs --port=8008
```
