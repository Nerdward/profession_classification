# Profession Classification

## Data Cleaning
After some data exploration, I found out some of the comments
were repeated, so I removed the repetition. This greatly helped the data labeling model.

## Data Labelling
To label the dataset, I used Mistral 7B Instruct model to 
automatically label the data.

## Embedding
To improve the quality of the classification model, I used an embedding model to convert the sentences into vectors. I got a model from [huggingface](https://huggingface.co/BAAI/bge-large-en-v1.5).

## Model Building
I tried out Xgboost and SVC models.
The SVC beat the XGBoost model in its accuracy score (76%).
It also perfomed in other metrics like precision, recall and f1-score. I saved the model [here](./data/model.pkl)

## Inference
To make inference with the model
1. Make sure to download the embedding model
2. Run the code at the inference part of the notebook
