# DeepLearningforRobotics
Deep Learning Projects for Robotics Applications

Project: Comparison of performance between Vanilla CNN( 3 Layers) with YoloV8.


Testing steps:
1. Create a new Conda Env

conda create -n testenv python=3.7

2. Activate new Conda Env

conda activate testenv

3. Install requirements

pip install -r .\requirements.txt

4. Run test Script to test the fully trained model

python .\test.py 

5. Run train script to start training a new model(will not replace the existing model but will create a new file)

python .\train.py

6. Check performance of YOLOv8 Classifier trained on Imagenet
python .\yoloimageclassification.py

7. Check tensorboard
tensorboard --logdir=runs