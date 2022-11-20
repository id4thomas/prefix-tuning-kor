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


# Guide
* clone UnifiedSKG folder from https://github.com/HKUNLP/UnifiedSKG
```
git clone https://github.com/HKUNLP/UnifiedSKG.git
```