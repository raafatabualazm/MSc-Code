import sys
sys.modules['gptqmodel'] = None

# Monkeypatch PreTrainedTokenizerFast to work around transformers 5.9.0 type mismatch with tokenizers 0.22.1 on AddedToken
try:
    import tokenizers
    from transformers import PreTrainedTokenizerFast, PreTrainedModel
    _old_add_tokens = PreTrainedTokenizerFast._add_tokens
    
    def _patched_add_tokens(self, new_tokens, special_tokens=False):
        dict_or_attr = lambda o, k, d: o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)
        conv_tokens = []
        for t in new_tokens:
            if isinstance(t, str):
                conv_tokens.append(t)
            else:
                conv_tokens.append(tokenizers.AddedToken(
                    dict_or_attr(t, 'content', str(t)),
                    single_word=dict_or_attr(t, 'single_word', False),
                    lstrip=dict_or_attr(t, 'lstrip', False),
                    rstrip=dict_or_attr(t, 'rstrip', False),
                    normalized=dict_or_attr(t, 'normalized', True),
                    special=dict_or_attr(t, 'special', False)
                ))
        if special_tokens:
            return self._tokenizer.add_special_tokens(conv_tokens)
        return self._tokenizer.add_tokens(conv_tokens)
        
    PreTrainedTokenizerFast._add_tokens = _patched_add_tokens
    
    # Support newer transformers quantizer tie check on older Salesforce model
    PreTrainedModel.all_tied_weights_keys = property(
        lambda self: {k: None for k in getattr(self, "_all_tied_weights_keys", [])} if isinstance(getattr(self, "_all_tied_weights_keys", None), (list, set)) else getattr(self, "_all_tied_weights_keys", {}),
        lambda self, val: setattr(self, "_all_tied_weights_keys", val)
    )
    
    # Patch CodeT5pEncoderDecoderModel.tie_weights dynamically inside PreTrainedModel.__init__
    _old_init = PreTrainedModel.__init__
    def _patched_init(self, *args, **kwargs):
        _old_init(self, *args, **kwargs)
        cls = self.__class__
        if cls.__name__ == "CodeT5pEncoderDecoderModel" and not getattr(cls, "_tie_weights_patched", False):
            old_tie_weights = cls.tie_weights
            def patched_tie_weights(self, *args, **kwargs):
                return old_tie_weights(self)
            cls.tie_weights = patched_tie_weights
            cls._tie_weights_patched = True
            print("Successfully patched CodeT5pEncoderDecoderModel.tie_weights dynamically!")
    PreTrainedModel.__init__ = _patched_init
except Exception as e:
    pass

from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
import torch

try:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        'Salesforce/codet5p-2b',
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map='auto'
    )
    print("Loaded model successfully!")
    for n, m in model.named_modules():
        if hasattr(m, "scale_attn"):
            print(f"Module: {n} | Class: {m.__class__.__name__} | scale_attn device: {m.scale_attn.device} | hasattr: {hasattr(m, 'scale_attn')} | is_tensor: {isinstance(m.scale_attn, torch.Tensor)}")
except Exception as e:
    import traceback
    traceback.print_exc()
