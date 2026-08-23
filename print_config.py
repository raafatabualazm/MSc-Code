from transformers import AutoConfig
config = AutoConfig.from_pretrained("Salesforce/codet5p-2b", trust_remote_code=True)
print(config)
