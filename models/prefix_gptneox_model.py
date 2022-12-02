import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXForCausalLM

class myGPTNeoXForCausalLM(GPTNeoXForCausalLM):
    # need to override prepare_inputs_for_generation - past_key_values kwargs key error
    def prepare_inputs_for_generation(self, input_ids, past_key_values = None, past=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape

        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        if past is None:
            past = past_key_values

        # attention_mask is applied after q*k^T
        # attention_mask should be length prefix_len + input_ids length
        if past is not None and attention_mask is not None:
            prefix_len = past[0][0].shape[-2]
            bsz = input_ids.shape[0]
            device = attention_mask.device
            prefix_attention_mask = torch.ones(bsz, prefix_len)
            attention_mask = attention_mask.detach().cpu()
            attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim = -1).to(device)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past, **model_kwargs}

class PrefixGPTNeoXLMHeadModel(nn.Module):
    main_input_name = "input_ids"
    def __init__(self, args):
        super().__init__()
        self.args = args
        """The prefix-tuning code"""

        self.preseqlen = args.prefix_sequence_length
        self.mid_dim = args.mid_dim

        print("prefix-tuning sequence length is {}.".format(self.preseqlen))

        # Load tokenizer and model.
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_location)

        self.pretrain_model = myGPTNeoXForCausalLM.from_pretrained(
            args.pretrained_location
        )

        self.config = self.pretrain_model.config

        # Parameter follows eluetherai/polyglot
        self.match_n_layer = self.config.num_hidden_layers
        self.match_n_head = self.config.num_attention_heads
        self.n_embd = self.config.hidden_size
        assert self.n_embd % self.match_n_head == 0
        self.match_n_embd = self.n_embd // self.match_n_head # huggingface BART's dim of kv need to be calculated

        if args.special_tokens:
            self.tokenizer.add_tokens([v for k, v in args.special_tokens])
            self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

        # Prefix related.
        self.register_buffer('input_tokens', torch.arange(self.preseqlen).long())

        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )
        self.dropout = nn.Dropout(args.prefix_dropout)
        
        #### PARAMETER FREEZE
        if self.args.freeze_plm:
            for param in self.pretrain_model.parameters():
                param.requires_grad = False
        if self.args.freeze_prefix:
            for param in self.wte.parameters():
                param.requires_grad = False
            for param in self.control_trans.parameters():
                param.requires_grad = False

    def get_prompt(self, bsz = None, sample_size=1):
        bsz = bsz * sample_size
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb

        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(self,
                input_ids = None,
                attention_mask = None,
                labels = None,
                **kwargs,
                ):
        bsz = input_ids.shape[0]

        past_prompt = self.get_prompt(
            bsz=bsz
        )

        # Make Prefix Attention mask
        if not attention_mask is None:
            device = attention_mask.device
            prefix_attention_mask = torch.ones(bsz, self.preseqlen)
            attention_mask = attention_mask.detach().cpu()
            attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim = -1).to(device)

        outputs = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_prompt,
            **kwargs,
        )

        return {'loss': outputs.loss, "logits": outputs.logits}

    def generate(self,
                 input_ids,
                 attention_mask = None,
                 **kwargs):

        bsz = input_ids.shape[0]

        past_prompt = self.get_prompt(
            bsz=bsz, sample_size=kwargs['num_beams']
        )

        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_prompt,
            use_cache=False, # Important! use_cache needs to be false, we provide our custom past
            **kwargs,
        )

        return generated_ids