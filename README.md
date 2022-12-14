# Neural_Network_Charity_Analysis

## Overview 
The Alphabet Soup’s business team has provided a [CSV](https://raw.githubusercontent.com/lkachury/Neural_Network_Charity_Analysis/main/charity_data.csv) file containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

- **EIN** and **NAME**—Identification columns
- **APPLICATION**_TYPE—Alphabet Soup application type
- **AFFILIATION**—Affiliated sector of industry
- **CLASSIFICATION**—Government organization classification
- **USE_CASE**—Use case for funding
- **ORGANIZATION**—Organization type
- **STATUS**—Active status
- **INCOME_AMT**—Income classification
- **SPECIAL_CONSIDERATIONS**—Special consideration for application
- **ASK_AMT**—Funding amount requested
- **IS_SUCCESSFUL**—Was the money used effectively

The features in the provided dataset will be used to help create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup. The purpose of this analysis is to use machine learning and neural networks to help the foundation predict where to make investments.

## Resources
### Software
- Python 3.7.13
- Conda 22.9.0
- Jupyter Notebook
- [TensorFlow](https://playground.tensorflow.org/#activation=sigmoid&batchSize=10&dataset=gauss&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=1&seed=0.10587&showTestData=false&discretize=true&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false&discretize_hide=true&regularization_hide=true&learningRate_hide=true&regularizationRate_hide=true&percTrainData_hide=true&showTestData_hide=true&noise_hide=true&batchSize_hide=true) Neural Network

### Data Source 
- Alphabet Soup Charity [charity_data](https://github.com/lkachury/Neural_Network_Charity_Analysis/blob/main/charity_data.csv) csv file

## Results
### Deliverable 1: Preprocessing Data for a Neural Network Model
Using Pandas and Scikit-Learn’s `StandardScaler()`, the dataset will be preprocessed in order to compile, train, and evaluate the neural network model later in Deliverable 2. The completed AlphabetSoupCharity Jupyter Notebook can be referenced [here](https://github.com/lkachury/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.ipynb).

The following preprocessing steps have been performed:
1. The `EIN` and `NAME` columns have been dropped: <br /> ![image](https://user-images.githubusercontent.com/108038989/200087693-28fecd68-aaff-475d-a557-d52f966d6ff5.png)

2. The columns with more than 10 unique values have been grouped together: <br /> ![image](https://user-images.githubusercontent.com/108038989/200087841-3ea68e50-949b-47d1-8dff-296ae84f1e94.png)

3. The categorical variables have been encoded using one-hot encoding: <br /> ![image](https://user-images.githubusercontent.com/108038989/200087907-aec3f0bc-27a3-4ed7-b732-2c87f0521f11.png)

4. The preprocessed data is split into features and target arrays: <br /> ![image](https://user-images.githubusercontent.com/108038989/200088021-75e8fc4e-5ec5-4199-979f-98f830dd0c90.png)

5. The preprocessed data is split into training and testing datasets: <br /> ![image](https://user-images.githubusercontent.com/108038989/200088050-2c9fd59e-aea6-4b88-97a2-37ad56cbad51.png)

6. The numerical values have been standardized using the `StandardScaler()` module: <br /> ![image](https://user-images.githubusercontent.com/108038989/200088084-64e82cd0-6f93-4d59-832a-294e8641051d.png)

#### **Data Preprocessing Results**
- **What variable(s) are considered the target(s) for your model?** <br /> **IS_SUCCESSFUL**—Was the money used effectively <br />
- **What variable(s) are considered to be the features for your model?** <br /> **APPLICATION**_TYPE—Alphabet Soup application type <br /> **AFFILIATION**—Affiliated sector of industry <br /> **CLASSIFICATION**—Government organization classification <br /> **USE_CASE**—Use case for funding <br /> **ORGANIZATION**—Organization type <br /> **STATUS**—Active status <br /> **INCOME_AMT**—Income classification <br /> **SPECIAL_CONSIDERATIONS**—Special consideration for application <br /> **ASK_AMT**—Funding amount requested <br />
- **What variable(s) are neither targets nor features, and should be removed from the input data?** <br /> **EIN** and **NAME**—Identification columns

### Deliverable 2: Compile, Train, and Evaluate the Model 
Using TensorFlow, a neural network, or deep learning model, will be designed to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. We’ll need to think about how many inputs there are before determining the number of neurons and layers in the model. Once this step is completed, we’ll compile, train, and evaluate the binary classification model to calculate the model’s loss and accuracy. The completed AlphabetSoupCharity Jupyter Notebook can be referenced [here](https://github.com/lkachury/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.ipynb).

The neural network model using Tensorflow Keras contains working code that performs the following steps:
1. The number of layers, the number of neurons per layer, and activation function are defined: <br /> ![image](https://user-images.githubusercontent.com/108038989/200126533-f9bc7393-83a3-4584-825e-8aec465a0eca.png)  

2. An output layer with an activation function is created: <br /> ![image](https://user-images.githubusercontent.com/108038989/200126549-8897706e-5d02-477a-8d28-07a517a33ad4.png)

3. There is an output for the structure of the model: <br /> ![image](https://user-images.githubusercontent.com/108038989/200126575-bf8c5345-3dc1-446b-a710-ea5f15020b80.png)

4. There is an output of the model’s loss and accuracy: <br /> ![image](https://user-images.githubusercontent.com/108038989/200126796-7064a3ef-e728-485e-89e6-3700e444e2ec.png)

5. The model's weights are saved every 5 epochs: <br /> ![image](https://user-images.githubusercontent.com/108038989/200126772-872edd64-e020-4d3d-b5c1-34c87d8097c8.png)

6. The results are saved to an [HDF5](https://github.com/lkachury/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.h5) file: <br /> ![image](https://user-images.githubusercontent.com/108038989/200126396-a420a5d2-c97f-4109-b8b4-873b91f95f51.png)

#### **Compiling, Training, and Evaluating the Model Results**
- **How many neurons, layers, and activation functions did you select for your neural network model, and why?** <br /> The neural network model in this deliverable consisted of two layers, the first with 80 neurons ans the second with 30 neurons. The two layers used the "relu" activation function and the output layer used the "sigmoid" activation feature.
- **Were you able to achieve the target model performance?** <br /> The model's accuracy was 72.7%, so it did not achieve the target model performance of 75%.

### Deliverable 3: Optimize the Model 
Using TensorFlow, the model will be optimized in order to achieve a target predictive accuracy higher than 75%. If we can't achieve an accuracy higher than 75%, then we'll need to make at least three attempts to do so. The completed AlphabetSoupCharity_Optimization Jupyter Notebook can be referenced [here](https://github.com/lkachury/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimization.ipynb).

The model was not succesfully optimized as the predictive accuracy did not increase to over 75%. Three attempts were made to increase model performance using the following steps:

- Attempt #1: 
        <br /> 1. Noisy variables are removed from features: <br /> ![image](https://user-images.githubusercontent.com/108038989/200187894-ab32cb3f-a23e-46e9-9526-9264c2ba1cb5.png)
        <br /> 2. and 3. Additional hidden layers are added and additional neurons are added to hidden layers: <br /> ![image](https://user-images.githubusercontent.com/108038989/200188621-b64fe3af-d7de-4a62-9fa0-e3b8ff3afe1e.png)
        <br /> 4. The activation function of hidden layers or output layers is changed for optimization: <br /> ![image](https://user-images.githubusercontent.com/108038989/200188744-97b207d2-a996-4453-ba29-4b48ab762492.png)
        <br /> 5. The model's weights are saved every 5 epochs: <br /> ![image](https://user-images.githubusercontent.com/108038989/200188897-160965dc-1fe3-45af-a14e-b019caf1e48d.png)
        <br /> 6. The results are saved to an [HDF5](https://github.com/lkachury/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimization.h5) file: <br /> ![image](https://user-images.githubusercontent.com/108038989/200188956-5522e984-1a84-483a-b26e-9d9707e631ea.png)

The model accuracy of this attempt: <br /> ![image](https://user-images.githubusercontent.com/108038989/200189013-128ec6ee-deee-4b57-a5f3-7a4d4678ba75.png)

- Attempt #2: 
        <br /> 1. Noisy variables are removed from features: Same as Attempt #1
        <br /> 2. and 3. Additional hidden layers are added and additional neurons are added to hidden layers: <br /> ![image](https://user-images.githubusercontent.com/108038989/200189843-6a0e3905-2648-4fcf-aaa7-fc8669d93cb1.png)
        <br /> 4. The activation function of hidden layers or output layers is changed for optimization: <br /> ![image](https://user-images.githubusercontent.com/108038989/200189865-cda3354c-9d6a-4d98-b0b3-99df10a56c94.png)
        <br /> 5. The model's weights are saved every 5 epochs: <br /> ![image](https://user-images.githubusercontent.com/108038989/200189904-5ff5da98-132f-43ae-b20a-3e3b8f9eedfd.png)
        <br /> 6. The results are saved to an [HDF5](https://github.com/lkachury/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimization.h5) file: Same as Attempt #1

The model accuracy of this attempt: <br /> ![image](https://user-images.githubusercontent.com/108038989/200189925-e539f668-9f4c-4ade-823b-2e5632c11840.png)
 
- Attempt #3: 
        <br /> 1. Noisy variables are removed from features: Same as Attempt #1
        <br /> 2. and 3. Additional hidden layers are added and additional neurons are added to hidden layers: <br /> ![image](https://user-images.githubusercontent.com/108038989/200189976-c1898023-c7fd-4a81-805c-bdb0e5b9f466.png)
        <br /> 4. The activation function of hidden layers or output layers is changed for optimization: <br /> ![image](https://user-images.githubusercontent.com/108038989/200189997-c860fac8-2dd9-42ea-9467-3f31ee13111c.png)
        <br /> 5. The model's weights are saved every 5 epochs: <br /> ![image](https://user-images.githubusercontent.com/108038989/200190019-18aac33f-f181-422e-bccd-53ae07631ef8.png)
        <br /> 6. The results are saved to an [HDF5](https://github.com/lkachury/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimization.h5) file: Same as Attempt #1

The model accuracy of this attempt: <br /> ![image](https://user-images.githubusercontent.com/108038989/200190033-c46bd159-4b2b-4778-9fa6-a343f97b181e.png)
 
## Summary
The purpose of this analysis was to utilize machine learning and neural networks to help the Alphabet Soup’s business team predict where to make successful investments based on data from previously funded organizations. The initial model accuracy was 72.7%, which failed to achieve the target model performance of 75%. Three additional attempts were made to optimize the model in order to achieve a target predictive accuracy higher than 75%, but these attempts yielded model accuracies of 72.6%, 72.7% and 72.5%. From these results, we can conclude that further optimization of the model would not yield increased accuracy and to continue to increase the number of layers and neurons could lead to overfitting. 

The use of a different model, such as Random Forest, could help solve the classification problem. Random forest models are similar to neural network models but they use decision trees and combine their multiple smaller outputs to make a more robust and accurate final classification (or regression) decision. Random forest classifiers are able to train on large datasets and achieve comparable predictive accuracy values in faster time and with less code, while the other model required more time to train on the same data points. 
