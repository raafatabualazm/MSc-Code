import os
import torch
import sys

# Ensure correct pathing
sys.path.insert(0, "/home/zeus/MSc-Code")

# Apply monkeypatch
from transformers import PreTrainedModel
PreTrainedModel.all_tied_weights_keys = property(
    lambda self: {k: None for k in getattr(self, "_all_tied_weights_keys", [])} if isinstance(getattr(self, "_all_tied_weights_keys", None), (list, set)) else getattr(self, "_all_tied_weights_keys", {}),
    lambda self, val: setattr(self, "_all_tied_weights_keys", val)
)

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
PreTrainedModel.__init__ = _patched_init

os.environ["GRAPH_DECODER_MODEL"] = "Salesforce/codet5p-2b"
os.environ["GRAPH_ENCODER_PEFT"] = "lora"
os.environ["GRAPH_DECODER_PEFT"] = "lora"

from scripts.training.graph_encoder_decoder_decompiler_v2_antigravity import GraphCodeBERTT5Seq2Seq

model = GraphCodeBERTT5Seq2Seq()
state = torch.load("/home/zeus/MSc-Code/artifacts/codet5p-2b_lora_enc_dec_r16_1e5/pytorch_model.bin", map_location="cpu")

missing, unexpected = model.load_state_dict(state, strict=False)
print("Missing keys count:", len(missing))
print("Unexpected keys count:", len(unexpected))

print("\nFirst 50 missing keys:")
for k in missing[:50]:
    print(k)
