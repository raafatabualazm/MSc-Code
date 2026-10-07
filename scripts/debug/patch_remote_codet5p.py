path = '/teamspace/studios/this_studio/.cache/huggingface/modules/transformers_modules/Salesforce/codet5p_hyphen_2b/0083d4d638746e6c9ee3dbd504e6dd68738e3c87/modeling_codet5p.py'
try:
    with open(path, 'r', encoding='utf-8') as f:
        code = f.read()
    target = 'if self.config.tie_encoder_decoder:'
    replacement = 'if getattr(self.config, "tie_encoder_decoder", False):'
    if target in code:
        code = code.replace(target, replacement)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(code)
        print('Successfully patched modeling_codet5p.py')
    else:
        print('Target not found in modeling_codet5p.py (maybe already patched?)')
except Exception as e:
    print(f'Error patching modeling_codet5p.py: {e}')
