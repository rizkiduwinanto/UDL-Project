from evaluate import load
from transformers import PreTrainedTokenizerBase
from sentence_transformers import SentenceTransformer
import spacy
import torch
from collections import defaultdict
from nltk.util import ngrams

def compute_perplexity(texts, model_id="gpt2"):
    torch.cuda.empty_cache() 
    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(predictions=texts, model_id=model_id, device='cuda')
    return results['mean_perplexity']

def compute_wordcount(texts):
    wordcount = load("word_count")
    wordcount = wordcount.compute(data=texts)
    return wordcount['unique_words']

def compute_mauve(texts_pred, texts_true, model_id="gpt2"):
    torch.cuda.empty_cache()
    mauve = load("mauve")
    results = mauve.compute(predictions=texts_pred, references=texts_true, featurize_model_name=model_id, max_text_length=256, device_id=0)
    return results.mauve, results.divergence_curve

def compute_diversity(texts):
    ngram_range = [2,3,4]

    tokenizer = spacy.load("en_core_web_sm").tokenizer
    token_list = []
    for sentence in texts:
        token_list.append([str(token) for token in tokenizer(sentence)])
    ngram_sets = {}
    ngram_counts = defaultdict(int)

    metrics = {}
    for n in ngram_range:
        ngram_sets[n] = set()
        for tokens in token_list:
            ngram_sets[n].update(ngrams(tokens, n))
            ngram_counts[n] += len(list(ngrams(tokens, n)))
        metrics[f'{n}gram_repetition'] = (1-len(ngram_sets[n])/ngram_counts[n])
    diversity = 1
    for val in metrics.values():
        diversity *= (1-val)
    metrics['diversity'] = diversity
    return metrics

def compute_memorization(texts_pred, texts_true, n=4):
    tokenizer = spacy.load("en_core_web_sm").tokenizer
    unique_four_grams = set()
    for sentence in texts_true:
        unique_four_grams.update(ngrams([str(token) for token in tokenizer(sentence)], n))

    total = 0
    duplicate = 0
    for sentence in texts_pred:
        four_grams = list(ngrams([str(token) for token in tokenizer(sentence)], n))
        total += len(four_grams)
        for four_gram in four_grams:
            if four_gram in unique_four_grams:
                duplicate += 1

    return duplicate/total