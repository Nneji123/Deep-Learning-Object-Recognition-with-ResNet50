## Deep Learning Object Recognition with ResNet50
A Streamlit app that uses the ResNet50 pre-trained model to classify objects from images.

[![Language](https://img.shields.io/badge/Python-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Framework](https://img.shields.io/badge/Streamlit-darkred.svg?style=flat&logo=streamlit&logoColor=white )](http://www.streamlit.com)
![hosted](https://img.shields.io/badge/Streamlit-Cloud-DC143C?style=flat&logo=streamlit&logoColor=white)
![build](https://img.shields.io/badge/build-passing-brightgreen.svg?style=flat)
![](https://img.shields.io/github/repo-size/Nneji123/Deep-Learning-Object-Recognition-with-ResNet50)

## About
A residual neural network (ResNet) is an artificial neural network (ANN). It is a gateless or open-gated variant of the HighwayNet, the first working very deep feedforward neural network with hundreds of layers, much deeper than previous neural networks.

ResNet-50 is a convolutional neural network that is 50 layers deep. You can load a pretrained version of the network trained on more than a million images from the ImageNet database. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals.

## Preview
![object](https://user-images.githubusercontent.com/101701760/171369253-39e0ec51-1613-4adb-8f39-5eb9e3da40da.gif)

## Requirements
To run a demo do the following:
1. Clone the repository.
2. Install the requirements from the requirements.txt file:
```
pip install -r requirements.txt
```
3. Then from your command line run:
```
streamlit run streamlit_app.py
```
Then you can view the site on your local server.

## Deployment
The app was made and deployed with streamlit and streamlit cloud. 
Streamlit is an open source app framework in Python language. It helps us create web apps for data science and machine learning in a short time. It is compatible with major Python libraries such as scikit-learn, Keras, PyTorch, SymPy(latex), NumPy, pandas, Matplotlib etc.

The live link can be seen here:
https://share.streamlit.io/nneji123/deep-learning-object-recognition-with-resnet50/main
