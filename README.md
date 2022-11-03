# Neural_Network_Charity_Analysis

## Overview 
The Alphabet Soup’s business team has provided a CSV file containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special consideration for application
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively

The features in the provided dataset will be used to help create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup. The purpose of this analysis is to use machine learning and neural networks to help the foundation predict where to make investments.

## Resources
### Software
- Python 3.7.13
- Conda 22.9.0
- Jupyter Notebook
- [TensorFlow](https://playground.tensorflow.org/#activation=sigmoid&batchSize=10&dataset=gauss&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=1&seed=0.10587&showTestData=false&discretize=true&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false&discretize_hide=true&regularization_hide=true&learningRate_hide=true&regularizationRate_hide=true&percTrainData_hide=true&showTestData_hide=true&noise_hide=true&batchSize_hide=true) Neural Network

### Data Source 
- Alphabet Soup Charity [charity_data]() csv file

## Results
### Deliverable 1: Preprocessing Data for a Neural Network Model
Using Pandas and Scikit-Learn’s `StandardScaler()`, the dataset will be preprocessed in order to compile, train, and evaluate the neural network model later in Deliverable 2. 

The following preprocessing steps have been performed:
1. The `EIN` and `NAME` columns have been dropped:

2. The columns with more than 10 unique values have been grouped together:

3. The categorical variables have been encoded using one-hot encoding:

4. The preprocessed data is split into features and target arrays:

5. The preprocessed data is split into training and testing datasets:

6. The numerical values have been standardized using the `StandardScaler()` module:


### Deliverable 2: Compile, Train, and Evaluate the Model 
Using TensorFlow, a neural network, or deep learning model, will be designed to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. We’ll need to think about how many inputs there are before determining the number of neurons and layers in the model. Once this step is completed, we’ll compile, train, and evaluate the binary classification model to calculate the model’s loss and accuracy.

The neural network model using Tensorflow Keras contains working code that performs the following steps:
1. The number of layers, the number of neurons per layer, and activation function are defined:

2. An output layer with an activation function is created:

3. There is an output for the structure of the model:

4. There is an output of the model’s loss and accuracy:

5. The model's weights are saved every 5 epochs:

6. The results are saved to an HDF5 file:

### Deliverable 3: Optimize the Model 
Using TensorFlow, the model will be optimize in order to achieve a target predictive accuracy higher than 75%. If we can't achieve an accuracy higher than 75%, then we'll need to make at least three attempts to do so.

The model is optimized, and the predictive accuracy is increased to over 75%, or there is working code that makes three attempts to increase model performance using the following steps:
1. Noisy variables are removed from features:

2. Additional neurons are added to hidden layers:

3. Additional hidden layers are added:

4. The activation function of hidden layers or output layers is changed for optimization:

5. The model's weights are saved every 5 epochs:

6. The results are saved to an HDF5 file:


## Summary
Overview of the analysis: Explain the purpose of this analysis.

Results: Using bulleted lists and images to support your answers, address the following questions.

1. Data Preprocessing
    - What variable(s) are considered the target(s) for your model?
    - What variable(s) are considered to be the features for your model?
    - What variable(s) are neither targets nor features, and should be removed from the input data?

2. Compiling, Training, and Evaluating the Model
    - How many neurons, layers, and activation functions did you select for your neural network model, and why?
    - Were you able to achieve the target model performance?
    - What steps did you take to try and increase model performance?

Summary: Summarize the overall results of the deep learning model. 
Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.
There is a recommendation on using a different model to solve the classification problem, and justification
