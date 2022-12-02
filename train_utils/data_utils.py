import torch
import numpy as np

def batch_tokenize_preprocess_decoder(batch, tokenizer, max_length):
    source, target = batch["source"], batch["target"]

    # For GPT-2
    
    # input_sents = ['<|startoftext|>'+  s + '<|endoftext|><|startoftext|>' + t + '<|endoftext|>' for s,t in zip(source, target)]
    input_sents = ['<s>'+  s + ' [A] ' + t + '</s>' for s,t in zip(source, target)]
    # input_sents = [s + ' [질문] ' + t for s,t in zip(source, target)]
    tokenized = tokenizer(input_sents, 
                                 truncation=True, 
                                 max_length=max_length, 
                                 padding="max_length", add_special_tokens = True)
      
    # batch = {k: v for k, v in tokenized.items()}
    batch = {"input_ids": tokenized["input_ids"]}

    # Ignore padding in the loss (-100 is ignored) - Masking
    # batch["labels"] = [
    #     [-100 if token == tokenizer.pad_token_id else token for token in l]
    #     for l in tokenized["input_ids"]
    # ]

    # Sentence too
    batch["source"] = source
    batch["target"] = target
    return batch
