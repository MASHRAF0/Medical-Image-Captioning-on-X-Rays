# Medical Image Captioning on X-Rays

## Description

The objective of this study is to classify medical images using the Convolutional Neural Network(CNN) Model.
CNN model with a well-processed dataset of medical images. This model can be used to classify medical images based on categories provided as per the training dataset.

![Medical](https://www.ucsfhealth.org/-/media/project/ucsf/ucsf-health/medical-tests/hero/x-ray-skeleton-2x.jpg?h=1112&w=2880&hash=CFA177B5092DFF1B3AC225908DF7C476)

## About the dataset

🏆 This dataset was developed in 2017 by Arturo Polanco Lozano. It is also known as the MedNIST dataset for radiology and medical imaging. For the preparation of this dataset, images have been gathered from several datasets, namely, TCIA, the RSNA Bone Age Challange, and the NIH Chest X-ray dataset.

This dataset contains 58954 medical images belonging to 6 categories — AbdomenCT(10000 images), HeadCT(10000 images), Hand(10000 images), CXR(10000 images), CXR(10000 images), BreastMRI(8954 images), ChestCT(10000 images).

## Libraries & Packages

![numpy](https://img.shields.io/badge/Numpy-%25100-blue)
![pandas](https://img.shields.io/badge/Pandas-%25100-brightgreen)
![ScikitLearn](https://img.shields.io/badge/ScikitLearn-%25100-red)
![Keras](https://img.shields.io/badge/Keras-100-brightgreen)
![Tensorflow](https://img.shields.io/badge/tensorflow-100-red)


## Requirements

- Python 3.x
- pandas
- scikit-learn
- random
- shutil
- os
- ImageDataGenerator
- flow_from_directory
- Deep Learning CNN

## Project Steps
- Prepare and preprocess dataset, including splitting it into training and testing sets.
- Import the necessary libraries, including Keras and any required layers and models.
- Define the architecture o CNN model, including the number of layers, the types of layers (e.g. convolutional, pooling, fully connected), and the number of units in each layer.
- Compile the model by specifying the optimizer, loss function, and metrics.
- Train the model on the training data, specifying the batch size and number of epochs.
- Evaluate the model on the testing data to determine its performance.
- Use the trained model to make predictions on new data.


## Model

The model used in this project is CNN Model Architecure.

## Conclusion

A CNN model for medical image captioning on chest X-rays would involve using a CNN to extract features from the X-ray images, and then using those features to generate captions. The CNN would be trained on the X-ray images and their corresponding captions to learn to identify patterns and features in the images relevant to the diagnosis or condition depicted, and to generate captions based on those features. The architecture would consist of one or multiple convolutional layers for feature extraction, followed by one or multiple fully connected layers for classification, and a final layer to produce the captions. The model would be trained end-to-end with the aim of minimising the difference between predicted captions and ground truth captions.



