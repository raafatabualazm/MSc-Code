import json
from transformers import AutoTokenizer

def count_tokens_in_jsonl(filepath: str, model_name: str = "deepseek-ai/DeepSeek-V3.2"):
    """
    Count tokens in a JSONL file using Qwen tokenizer for reasoning, assembly, and source fields.
    
    Args:
        filepath: Path to the JSONL file
        model_name: Hugging Face model name for the tokenizer
    """
    # Load the Qwen tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Initialize counters
    total_reasoning_tokens = 0
    total_assembly_tokens = 0
    total_source_tokens = 0
    total_combined_tokens = 0
    
    num_samples = 0
    
    # Stats for individual samples
    max_reasoning_tokens = 0
    max_assembly_tokens = 0
    max_source_tokens = 0
    max_combined_tokens = 0
    
    print(f"\nProcessing file: {filepath}")
    print("-" * 60)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num} due to JSON error: {e}")
                continue
            
            # Get fields (with defaults for missing fields)
            reasoning = data.get('reasoning', '')
            assembly = data.get('assembly', '')
            source = data.get('source', '')
            
            # Count tokens for each field
            reasoning_tokens = len(tokenizer.encode(reasoning)) if reasoning else 0
            assembly_tokens = len(tokenizer.encode(assembly)) if assembly else 0
            source_tokens = len(tokenizer.encode(source)) if source else 0
            combined_tokens = reasoning_tokens + assembly_tokens + source_tokens
            
            # Update totals
            total_reasoning_tokens += reasoning_tokens
            total_assembly_tokens += assembly_tokens
            total_source_tokens += source_tokens
            total_combined_tokens += combined_tokens
            
            # Update max values
            max_reasoning_tokens = max(max_reasoning_tokens, reasoning_tokens)
            max_assembly_tokens = max(max_assembly_tokens, assembly_tokens)
            max_source_tokens = max(max_source_tokens, source_tokens)
            max_combined_tokens = max(max_combined_tokens, combined_tokens)
            
            num_samples += 1
            
            # Print progress every 100 samples
            if num_samples % 100 == 0:
                print(f"Processed {num_samples} samples...")
    
    # Calculate averages
    avg_reasoning = total_reasoning_tokens / num_samples if num_samples > 0 else 0
    avg_assembly = total_assembly_tokens / num_samples if num_samples > 0 else 0
    avg_source = total_source_tokens / num_samples if num_samples > 0 else 0
    avg_combined = total_combined_tokens / num_samples if num_samples > 0 else 0
    
    # Print results
    print("\n" + "=" * 60)
    print("TOKEN COUNT RESULTS")
    print("=" * 60)
    print(f"File: {filepath}")
    print(f"Tokenizer: {model_name}")
    print(f"Total samples: {num_samples}")
    print("-" * 60)
    
    print("\n📊 TOTAL TOKENS:")
    print(f"  Reasoning:  {total_reasoning_tokens:>12,}")
    print(f"  Assembly:   {total_assembly_tokens:>12,}")
    print(f"  Source:     {total_source_tokens:>12,}")
    print(f"  Combined:   {total_combined_tokens:>12,}")
    
    print("\n📈 AVERAGE TOKENS PER SAMPLE:")
    print(f"  Reasoning:  {avg_reasoning:>12,.1f}")
    print(f"  Assembly:   {avg_assembly:>12,.1f}")
    print(f"  Source:     {avg_source:>12,.1f}")
    print(f"  Combined:   {avg_combined:>12,.1f}")
    
    print("\n🔝 MAXIMUM TOKENS (single sample):")
    print(f"  Reasoning:  {max_reasoning_tokens:>12,}")
    print(f"  Assembly:   {max_assembly_tokens:>12,}")
    print(f"  Source:     {max_source_tokens:>12,}")
    print(f"  Combined:   {max_combined_tokens:>12,}")
    
    print("=" * 60)
    
    return {
        'num_samples': num_samples,
        'total': {
            'reasoning': total_reasoning_tokens,
            'assembly': total_assembly_tokens,
            'source': total_source_tokens,
            'combined': total_combined_tokens
        },
        'average': {
            'reasoning': avg_reasoning,
            'assembly': avg_assembly,
            'source': avg_source,
            'combined': avg_combined
        },
        'max': {
            'reasoning': max_reasoning_tokens,
            'assembly': max_assembly_tokens,
            'source': max_source_tokens,
            'combined': max_combined_tokens
        }
    }


if __name__ == "__main__":
    # Count tokens in data/intermediate/dart_synth.jsonl
    results = count_tokens_in_jsonl("data/intermediate/dart_synth.jsonl")

