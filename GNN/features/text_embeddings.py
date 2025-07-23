from transformers import BertTokenizer, BertModel
import torch

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained('bert-base-uncased')

wires = {
    "wire1": ["red_wire", "on_table"],
    "wire2": ["blue_wire", "on_table"],
    "wire3": ["black_wire", "inserted"]
}

llm_out = "locked-red_wire-terminal_3"

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
BERTmodel = BertModel.from_pretrained('bert-base-uncased')

prompt_tokens = tokenizer(llm_out, return_tensors='pt')

class_tokens = {}
state_tokens = {}

for key,value in wires.items():
    class_tokens[key] = tokenizer(value[0], return_tensors='pt')
    state_tokens[key] =tokenizer(value[1], return_tensors='pt')
with torch.no_grad():
    prompt_embedding = BERTmodel(**prompt_tokens)
    prompt_last_hidden = prompt_embedding.last_hidden_state
    embedding = torch.mean(prompt_last_hidden, dim=1).numpy()
    print(prompt_last_hidden.shape)
    print(embedding.shape)
