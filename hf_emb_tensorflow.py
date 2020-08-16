from transformers import AutoTokenizer, TFAutoModel
import tensorflow

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    
    input_mask_expanded = tensorflow.expand_dims(attention_mask, axis=2)
    input_mask_expanded = tensorflow.broadcast_to(input_mask_expanded, token_embeddings.shape)
    input_mask_expanded = tensorflow.cast(input_mask_expanded, tensorflow.float32)
    
    sum_embeddings = tensorflow.math.reduce_sum(token_embeddings * input_mask_expanded, axis=1)
    summed_input_mask_expanded = tensorflow.math.reduce_sum(input_mask_expanded, axis=1)
    
    # we want to clip only values which are less than min (similar to pytorch code).
    # As tensorflow.clip_by_value expects max values as well, we'll set this value to max of array.
    # As a result, returned tensor would be clipped values which are less than min not max
    clip_value_min = 1e-9
    clip_value_max = tensorflow.reduce_max(summed_input_mask_expanded, keepdims=True, axis=1)
    sum_mask = tensorflow.clip_by_value(summed_input_mask_expanded, clip_value_min=clip_value_min,
                                       clip_value_max=clip_value_max)
    
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
model = TFAutoModel.from_pretrained("distilbert-base-uncased")

# max_length depends on choosen network, it can't be greater than the network's max_length.
# max_length=512 means 512 word pieces which are around 400-500 english words.
# max_length=128(word pieces) is used most of the time.
encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='tf')

model_output = model(encoded_input)
print(model_output[0].shape) # (Total_sentences, tokens, tokens_dim)
    
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
print(sentence_embeddings.shape) #(Total_sentences, tokens_dim)