import sys
from transformers import AutoModelForSeq2SeqLM

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

model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5p-2b", trust_remote_code=True)
print("Keys in state_dict:")
for k in list(model.state_dict().keys())[:100]:
    print(k)
