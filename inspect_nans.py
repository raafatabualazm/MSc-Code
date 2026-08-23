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

from transformers import AutoTokenizer
from scripts.evaluation.graph_inference_antigravity import GraphInferenceModel, build_blocks, load_jsonl
from models.graphcodebert_tensor_builder import GraphCodeBERTTensorBuilder
from scripts.data.dfg_extractor import LightweightDFGExtractor

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder_tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base", trust_remote_code=True)

tensor_builder = GraphCodeBERTTensorBuilder(encoder_tokenizer, max_seq_len=512)
dfg_extractor = LightweightDFGExtractor()

print("Loading model...")
model = GraphInferenceModel("Salesforce/codet5p-2b").to(device)
state = torch.load("/home/zeus/MSc-Code/artifacts/codet5p-2b_lora_enc_dec_r16_1e5/pytorch_model.bin", map_location=device)
model.decompiler.load_state_dict(state, strict=False)
model.eval()

# Load a test set record
rows = load_jsonl("/home/zeus/MSc-Code/data/datasets/test-set.jsonl")
row = rows[0]

block_tensors, graph_data = build_blocks(row, tensor_builder, dfg_extractor)

print("\n--- Checking Embeddings for NaNs ---")
embeddings = []
for block in block_tensors:
    output = model.decompiler.local_encoder(
        input_ids=block['input_ids'].to(device),
        attention_mask=block['attention_mask'].to(device),
        position_ids=block['position_ids'].to(device),
        token_type_ids=block['token_type_ids'].to(device),
    )
    embeddings.append(output.squeeze(0))
node_states = torch.stack(embeddings)
print("node_states: min", node_states.min().item(), "max", node_states.max().item(), "has_nan", torch.isnan(node_states).any().item())

pooled, encoder_attention_mask = model.decompiler.graph_encoder(
    node_states,
    graph_data.edge_index.to(device) if graph_data.edge_index is not None else None,
    graph_data.edge_attr.to(device) if graph_data.edge_attr is not None else None,
    list_of_B_i=None,
    region_ids=graph_data.region_id.to(device),
)
print("pooled: min", pooled.min().item(), "max", pooled.max().item(), "has_nan", torch.isnan(pooled).any().item())

projected = model.decompiler.projection(pooled)
print("projected: min", projected.min().item(), "max", projected.max().item(), "has_nan", torch.isnan(projected).any().item())
