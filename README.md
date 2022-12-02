# prefix-tuning-kor
Prefix Tuning for Korean LMs

# Environment
* transformers==4.19.2
* torch==1.14.0.dev20221029

# Models Code
* Prefix*Model uses implementation from https://github.com/HKUNLP/UnifiedSKG
    * "models/modeling_bart.py" from https://github.com/HKUNLP/UnifiedSKG/blob/main/models/prompt/modeling_bart.py
        * modified to prevent error "add_code_sample_docstrings() got an unexpected keyword argument 'tokenizer_class'"
    * base.py from https://github.com/HKUNLP/UnifiedSKG/blob/main/models/unified/base.py
    * PrefixModel follows https://github.com/HKUNLP/UnifiedSKG/blob/main/models/unified/prefixtuning.py
        * Description/Knowledge parts are removed
    * "utils/args_utils.py" Args from https://github.com/HKUNLP/UnifiedSKG/blob/main/utils/configue.py


* Decoder models (GPT2, GPT-NEO-X) have modified "prepare_inputs_for_generation" function
    * resizes attention mask to input_ids_len -> "prefix_len + input_ids_len"
    * reason: attention_mask is applied after q*k^T inside Attention class
        * -> attention_weight + attention_mask
        * q*k^T has prefix_len+input_ids_len as shape[-1]


# Guide
* clone UnifiedSKG folder from https://github.com/HKUNLP/UnifiedSKG
```
git clone https://github.com/HKUNLP/UnifiedSKG.git
```

# Training Example
* Trained 1.3B GPT-Neo-X Model ("EleutherAI/polyglot-ko-1.3b") using open chatbot data
    * Data from https://github.com/songys/Chatbot_data
* 0_data_preprocess.ipynb
    * Split data into Train/Val/Test
* 1_train_gpt_neox.ipynb
    * Train prefix weights using huggingface Trainer
* 2_generate_gptneox.ipynb
    * generation example (comparison with baseline)