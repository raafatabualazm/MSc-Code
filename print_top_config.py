import tokenizers
from transformers import PreTrainedTokenizerFast
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

from transformers import AutoConfig, AutoTokenizer
config = AutoConfig.from_pretrained("Salesforce/codet5p-770m", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-770m", trust_remote_code=True)

print("=== Top level config ===")
print("eos_token_id:", getattr(config, "eos_token_id", None))
print("pad_token_id:", getattr(config, "pad_token_id", None))
print("decoder_start_token_id:", getattr(config, "decoder_start_token_id", None))

print("\n=== Tokenizer info ===")
print("bos_token:", tokenizer.bos_token, tokenizer.bos_token_id)
print("eos_token:", tokenizer.eos_token, tokenizer.eos_token_id)
print("pad_token:", tokenizer.pad_token, tokenizer.pad_token_id)
