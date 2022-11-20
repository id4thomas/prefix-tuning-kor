import torch
from torch import nn
from transformers import AutoTokenizer, GPT2LMHeadModel

import sys
import os
sys.path.append(os.pardir)
from UnifiedSKG.models.unified.base import PushToHubFriendlyModel

class PrefixGPT2LMHeadModel(PushToHubFriendlyModel):
    main_input_name = "input_ids"
    def __init__(self, args):
        super().__init__()
        self.args = args
        """The prefix-tuning code"""

        self.preseqlen = args.prefix_sequence_length
        self.mid_dim = args.mid_dim

        print("prefix-tuning sequence length is {}.".format(self.preseqlen))

        # Load tokenizer and model.
        # self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_location, use_fast=False)
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_location)
        self.pretrain_model = GPT2LMHeadModel.from_pretrained(
            args.pretrained_location
        )
        self.config = self.pretrain_model.config

        # Parameter follows skt/kogpt2-base-v2
        self.match_n_layer = self.config.n_layer
        self.match_n_head = self.config.n_head
        self.n_embd = self.config.n_embd
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
        # result = []
        # for i, key_val in enumerate(past_key_values):
        #     temp = dict()
        #     temp["decoder_prompt"] = {
        #         "prev_key": key_val[0].contiguous(),
        #         "prev_value": key_val[1].contiguous(),
        #         "prev_key_padding_mask": torch.zeros(bsz, seqlen)
        #             .to(key_val.device)
        #             .bool()
        #         # bsz, preseqlen
        #     }
        #     result.append(temp)

        # return result

    def forward(self,
                input_ids,
                attention_mask,
                labels,
                **kwargs,
                ):
        bsz = input_ids.shape[0]

        past_prompt = self.get_prompt(
            bsz=bsz
        )

        loss = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_prompt,
        ).loss
        return {'loss': loss}

    def generate(self,
                 input_ids,
                 attention_mask,
                 **kwargs):

        bsz = input_ids.shape[0]

        past_prompt = self.get_prompt(
            bsz=bsz, sample_size=kwargs['num_beams']
        )
        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_prompt,
            use_cache=True,
            **kwargs,
        )

        return generated_ids