from transformers import AutoTokenizer, AutoModel
import torch

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    return sum_embeddings / sum_mask


sentences = [
    'This is an excellent movie',
    'The move was fantastic I like it',
    'Its a fantastic series',
    'It is a Wonderful movie',
    'I did not like the movie',
    'I will not recommend',
    'The acting is pathetic']

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
model.eval()

# max_length depends on choosen network, it can't be greater than the network's max_length.
# max_length=512 means 512 word pieces which are around 400-500 english words.
# max_length=128(word pieces) is used most of the time.
encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')

with torch.no_grad():
    model_output = model(**encoded_input)
    print(model_output[0].shape) # (Total_sentences, tokens, tokens_dim)
    
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
print(sentence_embeddings.shape) # (Total_sentences, tokens_dim)