#!/usr/bin/env python3
"""
Token-Matched Dataset Creator for Dart vs Swift Assembly-to-Code Translation
Matches examples from two JSONL files based on token counts in reasoning and assembly fields.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
from dataclasses import dataclass
from collections import defaultdict
import argparse


@dataclass
class TokenizedExample:
    """Stores an example with its token counts"""
    data: Dict
    assembly_tokens: int
    reasoning_tokens: int
    total_tokens: int
    index: int
    lang: str


class TokenMatcher:
    """Matches examples from two datasets based on token counts"""
    
    def __init__(self, tokenizer_path: str = "Qwen/Qwen3-4B-Thinking-2507"):
        """Initialize with tokenizer"""
        print(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        
    def tokenize_example(self, example: Dict, lang: str, index: int) -> TokenizedExample:
        """Tokenize an example and return token counts"""
        assembly = example.get("assembly", "")
        reasoning = example.get("reasoning", "")
        
        assembly_tokens = len(self.tokenizer.encode(assembly))
        reasoning_tokens = len(self.tokenizer.encode(reasoning))
        total_tokens = assembly_tokens + reasoning_tokens
        
        return TokenizedExample(
            data=example,
            assembly_tokens=assembly_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            index=index,
            lang=lang
        )
    
    def load_and_tokenize(self, filepath: Path, lang: str) -> List[TokenizedExample]:
        """Load JSONL file and tokenize all examples"""
        print(f"\nLoading {lang} data from {filepath}...")
        examples = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                data = json.loads(line.strip())
                tokenized = self.tokenize_example(data, lang, idx)
                examples.append(tokenized)
        
        print(f"Loaded {len(examples)} {lang} examples")
        return examples
    
    def find_best_matches(
        self, 
        dart_examples: List[TokenizedExample],
        swift_examples: List[TokenizedExample],
        tolerance: float = 0.1,
        max_diff: int = 50
    ) -> List[Tuple[TokenizedExample, TokenizedExample, float]]:
        """
        Find best matches between Dart and Swift examples.
        
        Args:
            dart_examples: List of tokenized Dart examples
            swift_examples: List of tokenized Swift examples
            tolerance: Relative tolerance for token count matching (0.1 = 10%)
            max_diff: Maximum absolute difference in total tokens
            
        Returns:
            List of (dart_example, swift_example, similarity_score) tuples
        """
        print(f"\nMatching examples with tolerance={tolerance}, max_diff={max_diff}...")
        
        matches = []
        used_swift_indices = set()
        
        # Sort Dart examples by total tokens for efficient matching
        dart_sorted = sorted(dart_examples, key=lambda x: x.total_tokens)
        
        for dart_ex in dart_sorted:
            best_match = None
            best_score = float('inf')
            
            for swift_ex in swift_examples:
                if swift_ex.index in used_swift_indices:
                    continue
                
                # Calculate differences
                assembly_diff = abs(dart_ex.assembly_tokens - swift_ex.assembly_tokens)
                reasoning_diff = abs(dart_ex.reasoning_tokens - swift_ex.reasoning_tokens)
                total_diff = abs(dart_ex.total_tokens - swift_ex.total_tokens)
                
                # Check absolute threshold
                if total_diff > max_diff:
                    continue
                
                # Check relative threshold for assembly
                assembly_rel_diff = assembly_diff / max(dart_ex.assembly_tokens, 1)
                if assembly_rel_diff > tolerance:
                    continue
                
                # Check relative threshold for reasoning
                reasoning_rel_diff = reasoning_diff / max(dart_ex.reasoning_tokens, 1)
                if reasoning_rel_diff > tolerance:
                    continue
                
                # Calculate combined score (lower is better)
                score = (assembly_diff + reasoning_diff + total_diff) / 3
                
                if score < best_score:
                    best_score = score
                    best_match = swift_ex
            
            if best_match:
                matches.append((dart_ex, best_match, best_score))
                used_swift_indices.add(best_match.index)
        
        print(f"Found {len(matches)} matched pairs")
        return matches
    
    def save_matched_dataset(
        self, 
        matches: List[Tuple[TokenizedExample, TokenizedExample, float]],
        output_dir: Path
    ):
        """Save matched datasets to separate files"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dart_output = output_dir / "dart_matched.jsonl"
        swift_output = output_dir / "swift_matched.jsonl"
        stats_output = output_dir / "matching_stats.json"
        
        dart_data = []
        swift_data = []
        stats = {
            "total_matches": len(matches),
            "token_statistics": {
                "assembly": {"dart": [], "swift": [], "diff": []},
                "reasoning": {"dart": [], "swift": [], "diff": []},
                "total": {"dart": [], "swift": [], "diff": []}
            }
        }
        
        print(f"\nSaving matched datasets to {output_dir}...")
        
        # Save Dart examples
        with open(dart_output, 'w', encoding='utf-8') as f:
            for dart_ex, swift_ex, score in matches:
                f.write(json.dumps(dart_ex.data, ensure_ascii=False) + '\n')
                dart_data.append(dart_ex)
                
                # Collect statistics
                stats["token_statistics"]["assembly"]["dart"].append(dart_ex.assembly_tokens)
                stats["token_statistics"]["assembly"]["swift"].append(swift_ex.assembly_tokens)
                stats["token_statistics"]["assembly"]["diff"].append(
                    abs(dart_ex.assembly_tokens - swift_ex.assembly_tokens)
                )
                
                stats["token_statistics"]["reasoning"]["dart"].append(dart_ex.reasoning_tokens)
                stats["token_statistics"]["reasoning"]["swift"].append(swift_ex.reasoning_tokens)
                stats["token_statistics"]["reasoning"]["diff"].append(
                    abs(dart_ex.reasoning_tokens - swift_ex.reasoning_tokens)
                )
                
                stats["token_statistics"]["total"]["dart"].append(dart_ex.total_tokens)
                stats["token_statistics"]["total"]["swift"].append(swift_ex.total_tokens)
                stats["token_statistics"]["total"]["diff"].append(
                    abs(dart_ex.total_tokens - swift_ex.total_tokens)
                )
        
        print(f"Saved {len(dart_data)} Dart examples to {dart_output}")
        
        # Save Swift examples
        with open(swift_output, 'w', encoding='utf-8') as f:
            for dart_ex, swift_ex, score in matches:
                f.write(json.dumps(swift_ex.data, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(dart_data)} Swift examples to {swift_output}")
        
        # Calculate and save statistics
        for key in ["assembly", "reasoning", "total"]:
            for lang in ["dart", "swift", "diff"]:
                values = stats["token_statistics"][key][lang]
                stats["token_statistics"][key][f"{lang}_mean"] = float(np.mean(values))
                stats["token_statistics"][key][f"{lang}_std"] = float(np.std(values))
                stats["token_statistics"][key][f"{lang}_min"] = int(np.min(values))
                stats["token_statistics"][key][f"{lang}_max"] = int(np.max(values))
                stats["token_statistics"][key][f"{lang}_median"] = float(np.median(values))
        
        with open(stats_output, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"Saved statistics to {stats_output}")
        
        # Print summary
        self.print_statistics(stats)
    
    def print_statistics(self, stats: Dict):
        """Print matching statistics"""
        print("\n" + "="*60)
        print("MATCHING STATISTICS")
        print("="*60)
        print(f"Total matched pairs: {stats['total_matches']}")
        print()
        
        for key in ["assembly", "reasoning", "total"]:
            print(f"{key.upper()} Tokens:")
            print(f"  Dart:  {stats['token_statistics'][key]['dart_mean']:.1f} ± "
                  f"{stats['token_statistics'][key]['dart_std']:.1f} "
                  f"(range: {stats['token_statistics'][key]['dart_min']}-"
                  f"{stats['token_statistics'][key]['dart_max']})")
            print(f"  Swift: {stats['token_statistics'][key]['swift_mean']:.1f} ± "
                  f"{stats['token_statistics'][key]['swift_std']:.1f} "
                  f"(range: {stats['token_statistics'][key]['swift_min']}-"
                  f"{stats['token_statistics'][key]['swift_max']})")
            print(f"  Diff:  {stats['token_statistics'][key]['diff_mean']:.1f} ± "
                  f"{stats['token_statistics'][key]['diff_std']:.1f} "
                  f"(max: {stats['token_statistics'][key]['diff_max']})")
            print()
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Match Dart and Swift datasets based on token counts"
    )
    parser.add_argument(
        "--dart-file",
        type=Path,
        required=True,
        help="Path to Dart JSONL file"
    )
    parser.add_argument(
        "--swift-file",
        type=Path,
        required=True,
        help="Path to Swift JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("matched_datasets"),
        help="Output directory for matched datasets (default: matched_datasets)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen3-4B-Thinking-2507",
        help="Tokenizer model path (default: Qwen/Qwen3-4B-Thinking-2507)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help="Relative tolerance for token count matching (default: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--max-diff",
        type=int,
        default=50,
        help="Maximum absolute difference in total tokens (default: 50)"
    )
    
    args = parser.parse_args()
    
    # Initialize matcher
    matcher = TokenMatcher(args.tokenizer)
    
    # Load and tokenize datasets
    dart_examples = matcher.load_and_tokenize(args.dart_file, "Dart")
    swift_examples = matcher.load_and_tokenize(args.swift_file, "Swift")
    
    # Find matches
    matches = matcher.find_best_matches(
        dart_examples,
        swift_examples,
        tolerance=args.tolerance,
        max_diff=args.max_diff
    )
    
    if not matches:
        print("\nWarning: No matches found. Try increasing tolerance or max_diff.")
        return
    
    # Save matched datasets
    matcher.save_matched_dataset(matches, args.output_dir)
    
    print(f"\n✓ Complete! Matched datasets saved to {args.output_dir}")


if __name__ == "__main__":
    main()