# General Sentence Embeddings
**Extract Sentence Embeddings from Hugging Face pre-trained models.**

This repo contains code for both tensorflow and pytorch. We can extract sentence embeddings for our dataset using any pre-trained Hugging Face models.
Sometimes out of the box embeddings work or sometimes they won't.
If you want to train/finetune on your own dataset, checkout [sentence-transformers](https://github.com/UKPLab/sentence-transformers).

These can be used for any semantic similarity search tasks, clustering etc. 

## Dependencies
* tensorflow 2.0.0
* pytorch 1.6.0
* transformers 3.0.2

## Working
The code works in the following way
1) Load model and its respective tokenizer.
2) Tokenize our sentences
3) Get token embeddings
4) Convert token embeddings to single sentence embeddings<sup>[1]</sup>.

[1]. There are many techniques to convert token embeddings to sentence embeddings, but SOTA is mean pooling.

## Benchmarks
Benchmarks using [SentEval](https://github.com/facebookresearch/SentEval/) are coming Soon.

## Other repos for Sentence Embeddings
* [Gesen](https://github.com/Maluuba/gensen)
* [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
* [InferSent](https://github.com/facebookresearch/InferSent)
* [Skip-Thought](https://github.com/ryankiros/skip-thoughts)
* [SBert](https://github.com/BinWang28/SBERT-WK-Sentence-Embedding)
* [Universal Sentence Encoder](https://arxiv.org/abs/1803.11175)
 
## Note
**This repo is inspired by [sentence-transformers](https://github.com/UKPLab/sentence-transformers). The pytorch code is from their repo.**
