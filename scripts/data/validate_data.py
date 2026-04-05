#!/usr/bin/env python3
"""
Data Validation and Quality Check Utility
Validates JSONL files before matching and checks data quality
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np


class DataValidator:
    """Validates and analyzes JSONL datasets"""
    
    def __init__(self):
        self.required_fields = {'assembly', 'reasoning'}
        self.recommended_fields = {'filename', 'function', 'source', 'lang'}
    
    def validate_file(self, filepath: Path, lang: str) -> Tuple[bool, List[str], Dict]:
        """
        Validate a JSONL file and return validation results
        
        Returns:
            (is_valid, errors, statistics)
        """
        print(f"\n{'='*60}")
        print(f"Validating {lang} file: {filepath}")
        print(f"{'='*60}")
        
        errors = []
        stats = {
            'total_lines': 0,
            'valid_entries': 0,
            'empty_assembly': 0,
            'empty_reasoning': 0,
            'missing_fields': Counter(),
            'assembly_lengths': [],
            'reasoning_lengths': [],
            'source_lengths': []
        }
        
        if not filepath.exists():
            errors.append(f"File not found: {filepath}")
            return False, errors, stats
        
        # Validate each line
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                stats['total_lines'] += 1
                
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON - {e}")
                    continue
                
                # Check required fields
                valid_entry = True
                for field in self.required_fields:
                    if field not in data:
                        errors.append(f"Line {line_num}: Missing required field '{field}'")
                        stats['missing_fields'][field] += 1
                        valid_entry = False
                    elif not data[field] or not data[field].strip():
                        if field == 'assembly':
                            stats['empty_assembly'] += 1
                        elif field == 'reasoning':
                            stats['empty_reasoning'] += 1
                        errors.append(f"Line {line_num}: Empty {field} field")
                        valid_entry = False
                
                # Check recommended fields
                for field in self.recommended_fields:
                    if field not in data:
                        stats['missing_fields'][field] += 1
                
                if valid_entry:
                    stats['valid_entries'] += 1
                    stats['assembly_lengths'].append(len(data['assembly']))
                    stats['reasoning_lengths'].append(len(data['reasoning']))
                    if 'source' in data:
                        stats['source_lengths'].append(len(data['source']))
        
        # Print validation results
        print(f"\nValidation Results:")
        print(f"  Total lines: {stats['total_lines']}")
        print(f"  Valid entries: {stats['valid_entries']}")
        print(f"  Errors found: {len(errors)}")
        
        if stats['empty_assembly'] > 0:
            print(f"  Empty assembly fields: {stats['empty_assembly']}")
        if stats['empty_reasoning'] > 0:
            print(f"  Empty reasoning fields: {stats['empty_reasoning']}")
        
        if stats['missing_fields']:
            print(f"\n  Missing fields:")
            for field, count in stats['missing_fields'].most_common():
                field_type = "required" if field in self.required_fields else "recommended"
                print(f"    - {field} ({field_type}): {count} entries")
        
        # Print statistics
        if stats['valid_entries'] > 0:
            print(f"\n  Length Statistics (characters):")
            
            if stats['assembly_lengths']:
                print(f"    Assembly:")
                print(f"      Mean: {np.mean(stats['assembly_lengths']):.0f}")
                print(f"      Median: {np.median(stats['assembly_lengths']):.0f}")
                print(f"      Min: {np.min(stats['assembly_lengths'])}")
                print(f"      Max: {np.max(stats['assembly_lengths'])}")
            
            if stats['reasoning_lengths']:
                print(f"    Reasoning:")
                print(f"      Mean: {np.mean(stats['reasoning_lengths']):.0f}")
                print(f"      Median: {np.median(stats['reasoning_lengths']):.0f}")
                print(f"      Min: {np.min(stats['reasoning_lengths'])}")
                print(f"      Max: {np.max(stats['reasoning_lengths'])}")
            
            if stats['source_lengths']:
                print(f"    Source:")
                print(f"      Mean: {np.mean(stats['source_lengths']):.0f}")
                print(f"      Median: {np.median(stats['source_lengths']):.0f}")
                print(f"      Min: {np.min(stats['source_lengths'])}")
                print(f"      Max: {np.max(stats['source_lengths'])}")
        
        is_valid = len(errors) == 0 and stats['valid_entries'] > 0
        
        if is_valid:
            print(f"\n✓ File is valid and ready for matching")
        else:
            print(f"\n✗ File has validation issues")
        
        return is_valid, errors, stats
    
    def compare_datasets(self, dart_stats: Dict, swift_stats: Dict):
        """Compare statistics between Dart and Swift datasets"""
        print(f"\n{'='*60}")
        print("Dataset Comparison")
        print(f"{'='*60}")
        
        print(f"\nSize Comparison:")
        print(f"  Dart entries: {dart_stats['valid_entries']}")
        print(f"  Swift entries: {swift_stats['valid_entries']}")
        print(f"  Size ratio: {dart_stats['valid_entries'] / max(swift_stats['valid_entries'], 1):.2f}")
        
        print(f"\nLength Comparison (mean character counts):")
        
        if dart_stats['assembly_lengths'] and swift_stats['assembly_lengths']:
            dart_asm = np.mean(dart_stats['assembly_lengths'])
            swift_asm = np.mean(swift_stats['assembly_lengths'])
            print(f"  Assembly:")
            print(f"    Dart:  {dart_asm:.0f} chars")
            print(f"    Swift: {swift_asm:.0f} chars")
            print(f"    Ratio: {dart_asm / swift_asm:.2f}")
        
        if dart_stats['reasoning_lengths'] and swift_stats['reasoning_lengths']:
            dart_reas = np.mean(dart_stats['reasoning_lengths'])
            swift_reas = np.mean(swift_stats['reasoning_lengths'])
            print(f"  Reasoning:")
            print(f"    Dart:  {dart_reas:.0f} chars")
            print(f"    Swift: {swift_reas:.0f} chars")
            print(f"    Ratio: {dart_reas / swift_reas:.2f}")
        
        print(f"\n{'='*60}")
    
    def suggest_parameters(self, dart_stats: Dict, swift_stats: Dict):
        """Suggest matching parameters based on dataset statistics"""
        print(f"\nRecommended Matching Parameters:")
        print(f"{'='*60}")
        
        # Calculate size difference
        size_ratio = dart_stats['valid_entries'] / max(swift_stats['valid_entries'], 1)
        
        if abs(size_ratio - 1.0) > 0.5:
            print(f"  ⚠ Warning: Dataset sizes differ significantly (ratio: {size_ratio:.2f})")
            print(f"    Consider:")
            print(f"    - Using --max-samples to limit matches")
            print(f"    - Balancing dataset sizes before matching")
        
        # Suggest tolerance based on length variation
        if dart_stats['assembly_lengths'] and swift_stats['assembly_lengths']:
            dart_std = np.std(dart_stats['assembly_lengths'])
            swift_std = np.std(swift_stats['assembly_lengths'])
            avg_std = (dart_std + swift_std) / 2
            
            if avg_std > 1000:
                suggested_tolerance = 0.15
                suggested_max_diff = 100
            elif avg_std > 500:
                suggested_tolerance = 0.10
                suggested_max_diff = 50
            else:
                suggested_tolerance = 0.05
                suggested_max_diff = 30
            
            print(f"\n  Based on length variation:")
            print(f"    --tolerance {suggested_tolerance}")
            print(f"    --max-diff {suggested_max_diff}")
        
        # Suggest batch processing for large datasets
        total_size = dart_stats['valid_entries'] + swift_stats['valid_entries']
        if total_size > 10000:
            print(f"\n  ⚠ Large dataset detected ({total_size} total entries)")
            print(f"    Consider using batch_matcher.py for better memory efficiency:")
            print(f"    python batch_matcher.py --dart-file ... --swift-file ...")
        
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate and analyze JSONL datasets before matching"
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
        "--save-errors",
        type=Path,
        help="Save validation errors to file"
    )
    
    args = parser.parse_args()
    
    validator = DataValidator()
    
    # Validate Dart file
    dart_valid, dart_errors, dart_stats = validator.validate_file(args.dart_file, "Dart")
    
    # Validate Swift file
    swift_valid, swift_errors, swift_stats = validator.validate_file(args.swift_file, "Swift")
    
    # Compare datasets
    if dart_valid and swift_valid:
        validator.compare_datasets(dart_stats, swift_stats)
        validator.suggest_parameters(dart_stats, swift_stats)
    
    # Save errors if requested
    if args.save_errors and (dart_errors or swift_errors):
        with open(args.save_errors, 'w') as f:
            f.write("DART FILE ERRORS:\n")
            f.write("="*60 + "\n")
            for error in dart_errors:
                f.write(error + "\n")
            
            f.write("\n\nSWIFT FILE ERRORS:\n")
            f.write("="*60 + "\n")
            for error in swift_errors:
                f.write(error + "\n")
        
        print(f"\nErrors saved to {args.save_errors}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    if dart_valid and swift_valid:
        print("✓ Both files are valid and ready for matching")
        print("\nNext steps:")
        print("  python token_matcher.py --dart-file ... --swift-file ...")
    else:
        print("✗ Validation failed. Please fix the errors before matching.")
        if not dart_valid:
            print(f"  Dart file: {len(dart_errors)} errors")
        if not swift_valid:
            print(f"  Swift file: {len(swift_errors)} errors")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()