from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import torch
import streamlit as st
import time
from boolq_inference import single_predict
import pandas as pd

@st.cache(allow_output_mutation=True)
def load_model():
  tokenizer = AutoTokenizer.from_pretrained("roberta-base") 
  model = AutoModelForSequenceClassification.from_pretrained("boolq_model")
  return tokenizer, model


def input_validation(question, passage):
  #validate the inputs
  if not passage or passage.strip() == '':
    st.error("Invalid passage")
    st.stop()

  if not question or question.strip() == '':
    st.error("Invalid question")
    st.stop()

def compute_boolq():
  #title of web app
  st.title("Bool Question Answering")

  #load model
  tokenizer, model = load_model()
  with st.form("boolq form"):
    #take inputs from user
    passage = st.text_input("Enter passage")
    question = st.text_input("Enter question")

    

    if st.form_submit_button("Get Answer"):
      #validation
      input_validation(question, passage)
      with st.spinner("Please wait"):
          start_time = time.time()
          question, passage = question.strip(), passage.strip()
          result = single_predict(question, passage, tokenizer, model)
          end_time = time.time()
          
          st.success("Execution completed in {} seconds".format(round(end_time-start_time, 2)))
          yes_proba = result["Yes"]
          no_proba = result["No"]

          col1, col2 = st.columns(2)
          col1.metric("Yes probabiltity", yes_proba*100)
          col2.metric("No probability", no_proba*100)
          st.balloons()



  #displaying model information
  model_status = st.checkbox("Model Info")    
  if model_status:
    st.subheader("Model Parameters")
    model_info = {"model name": "roberta-base", "learning rate": "4e-6", "batch size": 32, "epochs": 20}
    model_info_df = pd.DataFrame(model_info, index=[1])
    st.dataframe(model_info_df)

    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    col1.image("model_info/accuracy.png", caption="model accuracy")
    col2.image("model_info/loss.png", caption="model loss")

if __name__ == "__main__":
  compute_boolq()

# while True:
#   context = input("Enter context: ")
#   question = input("enter question: ")
#   single_predict(question.strip(), context.strip())
