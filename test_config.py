from transformers import AutoConfig
models = [
    "Salesforce/codet5p-770m",
    "Salesforce/codet5p-2b",
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B"
]
for model in models:
    try:
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        print(f"SUCCESS {model} config load: {type(config)}")
    except Exception as e:
        print(f"FAILED {model} config load: {e}")
