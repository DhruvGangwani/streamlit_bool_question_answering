from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import torch


def single_predict(questions, passage, tokenizer, model):
  
  sequence = tokenizer.encode_plus(questions, passage, return_tensors="pt")['input_ids']
  
  logits = model(sequence)[0]
  probabilities = torch.softmax(logits, dim=1).detach().cpu().tolist()[0]
  proba_yes = round(probabilities[1], 2)
  proba_no = round(probabilities[0], 2)

  print(f"Question: {questions}, Yes: {proba_yes}, No: {proba_no}")
  return {"Yes": proba_yes, "No": proba_no}
