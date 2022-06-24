# The prediction of covid cases
 The dataset of 600 data was trained in RNN approach to build a prediction model for new covid cases in Malaysia.
 
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

# Project Title
The prediction of covid cases in Malaysia. 

## Description
In this project, Neural Network approach was used to conduct the time-series analysis project. The dataset consisting of numbers of covid cases and covid cases different clusters. The new_cases variable within 30 window size was trained for model development. The exploratory data analysis was conducted to remove the NaNs and interpolate the NaNs in the datasets. The MinMaxScaler was then performed on the target variable(new_cases) prior to the model creation and development steps. After the training has completed, the trained data was deployed to evaluate the model.

## Results and Deployment
In the pre-processing steps, the data was transformed and fitted in a few steps such as Min Max Scalling and etc. In the model development procedure, the model was created with additions of LSTM,Dense and dropout layers as shown in Model's architecture below.

![model](https://user-images.githubusercontent.com/107612253/175615845-e02326e8-0fb0-40d2-bb12-a06b4d606827.png)


The created model was trained with a batch size of 100 and epochs value of 150 to achieve an accuracy determined by MAPE and MAE as shown in the model's evaluation result.

![model evaluation](https://user-images.githubusercontent.com/107612253/175615988-83854a99-f1d0-4212-906f-68ba1761e0f3.jpg)

Since the problem is a regression problem, a linear activation function was used at the output layer. The mape and mse was used to determine the accuracy and loss of the model performance during training. The model evaluation showed that the model return 0.14% of mae error from passing in the predictions and the actual price in the dataset. Despite the additions of nodes in the dense layer, LSTM and dropout layers in the model’s creation, the model's performance improved insignificantly and visualized in the Tensorboard and To overcome the overfitting learning model, adding additional training data may help. Hence, the model can improve its accuracy by adding other layers such simpleRNN layer as well as increasing number of nodes in the LSTM layer would help. Besides, incrasing the window size (more than 60 days) would also help the model in predicting the new cases more accurately.

![Tensorboard](https://user-images.githubusercontent.com/107612253/175617716-2fd26afc-49f6-4ae4-a854-6e5d5ab6b980.jpg)
!Model deployment](https://user-images.githubusercontent.com/107612253/175617152-d41d5669-2be7-46ea-b393-a767774f286a.png)

## Acknowledgement
I would like to sincerely thank the Malaysia's Ministry of Health  for contributing the dataset which can be downloaded from [github](https://github.com/MoH-Malaysia/covid19-public).
