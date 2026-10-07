import torch
state = torch.load("/home/zeus/MSc-Code/artifacts/codet5p-2b_lora_enc_dec_r16_1e5/pytorch_model.bin", map_location="cpu")
print("Total keys in checkpoint:", len(state))
dec_keys = [k for k in state.keys() if "decoder" in k or "t5_model" in k]
print("Decoder/t5_model keys in checkpoint:", len(dec_keys))
for k in dec_keys[:50]:
    print(k)
