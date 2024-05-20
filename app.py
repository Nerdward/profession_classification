from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from typing import List
import torch
import torch.nn.functional as F
import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

@st.cache_resource
def load_artifacts():
    with open("model/model.pkl", "rb") as f:
        clf = pickle.load(f)

    encoder = LabelEncoder()
    encoder.classes_ = np.load('model/classes.npy',allow_pickle=True)

    return clf, encoder

def clean_comments(comment: str):
    list_comments = comment.split("|")
    unique_comments = list(dict.fromkeys(list_comments))

    return "".join(unique_comments)

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
    model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5')

    return model, tokenizer

def get_embeddings(texts: List[str], batch_size: int):
    all_embeddings = []
    print(f"Total number of records: {len(texts)}")
    print(f"Num batches: {(len(texts) // batch_size) + 1}")
    
    model, tokenizer = load_model()
    # Extract embeddings for the texts in batches
    for start_index in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[start_index:start_index + batch_size]

        # Generate tokens
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            model_output = model(**inputs)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]

        # Get the last hidden stated and pool them into a mean vector calculated across the sequence length dimension
        # This will reduce the output vector from [batch_size, sequence_length, hidden_layer_size]
        # to [batch_size, hidden_layer_size] thereby generating the embeddings for all the sequences in the batch
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        # Append to the embeddings list
        all_embeddings.extend(sentence_embeddings.tolist())

    return all_embeddings

def inference(df: pd.DataFrame):   
    df["comments"] = df["comments"].apply(clean_comments)
    
    test_embeddings = get_embeddings(texts=df["comments"].tolist(), batch_size=64)
    test_embeddings_df = pd.DataFrame(test_embeddings)
    
    clf, encoder = load_artifacts()
    y_pred = clf.predict(test_embeddings_df)
    # convert to labels
    y_pred_labels = encoder.inverse_transform(y_pred)
    
    y_pred = clf.predict(test_embeddings_df)
    # convert to labels
    y_pred_labels = encoder.inverse_transform(y_pred)
    
    df["labels"] = y_pred_labels
    return df

uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    predict = st.button("Predict")
    if predict:
        with st.spinner('Processing...'):
            df = inference(dataframe)

            st.dataframe(df)