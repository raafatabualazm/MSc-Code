"""
CodeBLEU metric implementation for code evaluation
Faithful implementation based on the original CodeBLEU paper:
"CodeBLEU: a Method for Automatic Evaluation of Code Synthesis"
https://arxiv.org/abs/2009.10297

Supports Dart and Swift languages using Tree-sitter
"""

import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Set, Union
import numpy as np
from tree_sitter import Language, Parser, Node
import tree_sitter_dart
import tree_sitter_swift

# Import BLEU score components
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.util import ngrams
import nltk

# Constants from the original paper
NGRAM_ORDER = 4  # Maximum n-gram order
DEFAULT_WEIGHTS = (0.25, 0.25, 0.25, 0.25)  # α, β, γ, δ for each component
KEYWORD_WEIGHT = 5.0  # Keywords get 5x weight in weighted BLEU


class CodeBLEUCalculator:
    """
    Faithful implementation of CodeBLEU metric from the paper.
    
    CodeBLEU = α·BLEU + β·BLEU_weight + γ·Match_ast + δ·Match_df
    where α = β = γ = δ = 0.25 by default
    """
    
    def __init__(self, lang: str, alpha: float = 0.25, beta: float = 0.25, 
                 gamma: float = 0.25, delta: float = 0.25):
        """
        Initialize CodeBLEU calculator.
        
        Args:
            lang: Programming language ('dart' or 'swift')
            alpha: Weight for BLEU score
            beta: Weight for weighted BLEU score
            gamma: Weight for AST match
            delta: Weight for dataflow match
        """
        self.lang = lang.lower()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        # Ensure weights sum to 1
        weight_sum = alpha + beta + gamma + delta
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")
        
        # Initialize Tree-sitter parser with correct API
        if self.lang == 'dart':
            self.language = Language(tree_sitter_dart.language())
        elif self.lang == 'swift':
            self.language = Language(tree_sitter_swift.language())
        else:
            raise ValueError(f"Unsupported language: {lang}")
        
        self.parser = Parser(self.language)
        
        # Language-specific keywords for weighted BLEU
        self.keywords = self._get_language_keywords()
        
    def _get_language_keywords(self) -> Set[str]:
        """
        Get language-specific keywords for weighted matching.
        Based on reserved words and important constructs.
        """
        # Dart keywords from the language specification
        dart_keywords = {
            'abstract', 'as', 'assert', 'async', 'await', 'break', 'case', 'catch',
            'class', 'const', 'continue', 'covariant', 'default', 'deferred', 'do',
            'dynamic', 'else', 'enum', 'export', 'extends', 'extension', 'external',
            'factory', 'false', 'final', 'finally', 'for', 'Function', 'get', 'hide',
            'if', 'implements', 'import', 'in', 'interface', 'is', 'late', 'library',
            'mixin', 'new', 'null', 'on', 'operator', 'part', 'required', 'rethrow',
            'return', 'set', 'show', 'static', 'super', 'switch', 'sync', 'this',
            'throw', 'true', 'try', 'typedef', 'var', 'void', 'while', 'with', 'yield'
        }
        
        # Swift keywords from the language specification
        swift_keywords = {
            'associatedtype', 'class', 'deinit', 'enum', 'extension', 'fileprivate',
            'func', 'import', 'init', 'inout', 'internal', 'let', 'open', 'operator',
            'private', 'protocol', 'public', 'rethrows', 'static', 'struct', 'subscript',
            'typealias', 'var', 'break', 'case', 'continue', 'default', 'defer', 'do',
            'else', 'fallthrough', 'for', 'guard', 'if', 'in', 'repeat', 'return',
            'switch', 'where', 'while', 'as', 'Any', 'catch', 'false', 'is', 'nil',
            'super', 'self', 'Self', 'throw', 'throws', 'true', 'try', '_',
            '#available', '#colorLiteral', '#column', '#else', '#elseif', '#endif',
            '#error', '#file', '#fileLiteral', '#function', '#if', '#imageLiteral',
            '#line', '#selector', '#sourceLocation', '#warning', 'associativity',
            'convenience', 'dynamic', 'didSet', 'final', 'get', 'infix', 'indirect',
            'lazy', 'left', 'mutating', 'none', 'nonmutating', 'optional', 'override',
            'postfix', 'precedence', 'prefix', 'Protocol', 'required', 'right', 'set',
            'Type', 'unowned', 'weak', 'willSet', 'async', 'await', 'actor'
        }
        
        return dart_keywords if self.lang == 'dart' else swift_keywords
    
    def tokenize_code(self, code: str, keep_comments: bool = False) -> List[str]:
        """
        Tokenize code following the paper's approach.
        
        Args:
            code: Source code string
            keep_comments: Whether to keep comments in tokenization
            
        Returns:
            List of tokens
        """
        if not keep_comments:
            # Remove single-line comments
            code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
            # Remove multi-line comments
            code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        # Tokenize: split on whitespace and punctuation, keeping punctuation as tokens
        # This matches the approach in the original implementation
        tokens = re.findall(r'\w+|[^\w\s]', code)
        
        # Filter empty tokens
        tokens = [t for t in tokens if t.strip()]
        
        return tokens
    
    def calc_bleu(self, references: List[List[str]], hypothesis: List[str], 
                  ngram_order: int = NGRAM_ORDER) -> float:
        """
        Calculate standard BLEU score.
        
        Args:
            references: List of reference token lists
            hypothesis: Hypothesis token list
            ngram_order: Maximum n-gram order (default 4)
            
        Returns:
            BLEU score
        """
        if not hypothesis:
            return 0.0
        
        # Use uniform weights for n-grams up to ngram_order
        weights = tuple([1.0 / ngram_order] * ngram_order)
        
        # Use smoothing to handle zero counts
        smoothing = SmoothingFunction().method4
        
        try:
            score = sentence_bleu(references, hypothesis, weights=weights,
                                smoothing_function=smoothing)
        except ZeroDivisionError:
            score = 0.0
            
        return score
    
    def calc_weighted_bleu(self, references: List[List[str]], hypothesis: List[str]) -> float:
        """
        Calculate weighted BLEU score where keywords have higher weight.
        Uses unigram precision with 5x weight for keywords as per the paper.
        
        Args:
            references: List of reference token lists
            hypothesis: Hypothesis token list
            
        Returns:
            Weighted BLEU score (unigram precision with keyword weighting)
        """
        if not hypothesis:
            return 0.0
        
        # Calculate weighted unigram matches
        hypothesis_tokens = Counter(hypothesis)
        
        # For each reference, find maximum matches
        max_matches = Counter()
        for reference in references:
            reference_tokens = Counter(reference)
            for token in hypothesis_tokens:
                max_matches[token] = max(max_matches[token], 
                                        min(hypothesis_tokens[token], reference_tokens[token]))
        
        # Calculate weighted precision
        weighted_matches = 0
        total_weights = 0
        
        for token, count in hypothesis_tokens.items():
            # Keywords get 5x weight as mentioned in the paper
            weight = 5.0 if token in self.keywords else 1.0
            weighted_matches += weight * max_matches[token]
            total_weights += weight * count
        
        if total_weights == 0:
            return 0.0
            
        # Return weighted unigram precision
        return weighted_matches / total_weights
    
    def get_ast_subtrees(self, node: Node, code_bytes: bytes, 
                        subtrees: Optional[List[Tuple[str, int, int]]] = None,
                        ignore_leaves: bool = True) -> List[Tuple[str, int, int]]:
        """
        Extract subtrees from AST for matching.
        
        Args:
            node: Tree-sitter node
            code_bytes: Original code as bytes
            subtrees: List to accumulate subtrees
            ignore_leaves: If True, ignore leaf nodes (as per paper)
            
        Returns:
            List of (node_type, start_byte, end_byte) tuples
        """
        if subtrees is None:
            subtrees = []
        
        # Only add non-leaf nodes when ignore_leaves is True
        if not ignore_leaves or len(node.children) > 0:
            subtrees.append((node.type, node.start_byte, node.end_byte))
        
        # Recursively process children
        for child in node.children:
            self.get_ast_subtrees(child, code_bytes, subtrees, ignore_leaves)
        
        return subtrees
    
    def calc_ast_match(self, reference: str, hypothesis: str) -> float:
        """
        Calculate syntactic AST match score as defined in the paper.
        Ignores leaf nodes and counts only internal AST nodes.
        
        Match_ast = count(hypothesis ∩ reference) / count(hypothesis)
        
        Args:
            reference: Reference code
            hypothesis: Generated code
            
        Returns:
            AST match score
        """
        try:
            ref_bytes = bytes(reference, 'utf8')
            hyp_bytes = bytes(hypothesis, 'utf8')
            
            ref_tree = self.parser.parse(ref_bytes)
            hyp_tree = self.parser.parse(hyp_bytes)
            
            # Extract subtrees, ignoring leaves as per paper
            ref_subtrees = self.get_ast_subtrees(ref_tree.root_node, ref_bytes, ignore_leaves=True)
            hyp_subtrees = self.get_ast_subtrees(hyp_tree.root_node, hyp_bytes, ignore_leaves=True)
            
            if not hyp_subtrees:
                return 0.0
            
            # Count matching subtrees based on node type (internal nodes only)
            ref_types = Counter([t[0] for t in ref_subtrees])
            hyp_types = Counter([t[0] for t in hyp_subtrees])
            
            # Calculate intersection
            match_count = 0
            for node_type, count in hyp_types.items():
                if node_type in ref_types:
                    match_count += min(count, ref_types[node_type])
            
            total_hyp = sum(hyp_types.values())
            
            return match_count / total_hyp if total_hyp > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def extract_dataflow_edges(self, tree: Node, code: str) -> Set[Tuple[str, str]]:
        """
        Extract data flow edges (variable dependencies) from code.
        Returns set of (source, target) pairs representing data flow.
        """
        edges = set()
        variables = {}  # Track variable definitions
        
        def traverse(node, in_assignment=False, assignment_target=None):
            """Traverse AST to extract data flow information."""
            
            # Handle different node types based on language constructs
            if node.type in ['identifier', 'simple_identifier']:
                var_name = code[node.start_byte:node.end_byte]
                
                if in_assignment and assignment_target is None:
                    # This is the target of an assignment
                    assignment_target = var_name
                    variables[var_name] = node
                elif assignment_target and var_name != assignment_target:
                    # This is a use in the RHS of an assignment
                    edges.add((var_name, assignment_target))
                elif not in_assignment and var_name in variables:
                    # Track usage in other contexts
                    pass
            
            # Handle assignment expressions
            elif node.type in ['assignment_expression', 'variable_declaration', 
                             'initialized_identifier', 'pattern_assignment']:
                # Process LHS first to get target
                if len(node.children) >= 2:
                    # First child is usually the target
                    target_node = node.children[0]
                    if target_node.type in ['identifier', 'simple_identifier']:
                        target = code[target_node.start_byte:target_node.end_byte]
                        variables[target] = target_node
                        
                        # Process RHS for dependencies
                        for i in range(1, len(node.children)):
                            traverse(node.children[i], True, target)
                        return
            
            # Handle function calls - track which variables are used
            elif node.type in ['call_expression', 'method_invocation']:
                # Extract function/method name and arguments
                for child in node.children:
                    if child.type == 'arguments':
                        for arg in child.children:
                            traverse(arg, False, None)
                    else:
                        traverse(child, in_assignment, assignment_target)
                return
            
            # Continue traversing children
            for child in node.children:
                traverse(child, in_assignment, assignment_target)
        
        try:
            traverse(tree, False, None)
        except Exception:
            pass
        
        return edges
    
    def calc_dataflow_match(self, reference: str, hypothesis: str) -> float:
        """
        Calculate semantic data-flow match score as defined in the paper.
        
        Match_df = count(hypothesis ∩ reference) / count(hypothesis)
        
        Args:
            reference: Reference code
            hypothesis: Generated code
            
        Returns:
            Data-flow match score
        """
        try:
            ref_tree = self.parser.parse(bytes(reference, 'utf8'))
            hyp_tree = self.parser.parse(bytes(hypothesis, 'utf8'))
            
            ref_edges = self.extract_dataflow_edges(ref_tree.root_node, reference)
            hyp_edges = self.extract_dataflow_edges(hyp_tree.root_node, hypothesis)
            
            if not hyp_edges:
                # If no data flow in hypothesis, check if reference also has none
                return 1.0 if not ref_edges else 0.0
            
            # Count matching edges
            matching_edges = ref_edges.intersection(hyp_edges)
            
            return len(matching_edges) / len(hyp_edges)
            
        except Exception:
            return 0.0
    
    def compute_codebleu(self, references: Union[str, List[str]], 
                        hypothesis: str) -> Dict[str, float]:
        """
        Compute CodeBLEU score following the paper's formula:
        CodeBLEU = α·BLEU + β·BLEU_weight + γ·Match_ast + δ·Match_df
        
        Args:
            references: Reference code(s) - can be string or list of strings
            hypothesis: Generated code
            
        Returns:
            Dictionary containing individual scores and final CodeBLEU
        """
        # Ensure references is a list
        if isinstance(references, str):
            references = [references]
        
        # Tokenize all codes
        ref_tokens_list = [self.tokenize_code(ref) for ref in references]
        hyp_tokens = self.tokenize_code(hypothesis)
        
        # 1. Calculate standard BLEU
        bleu_score = self.calc_bleu(ref_tokens_list, hyp_tokens)
        
        # 2. Calculate weighted BLEU
        weighted_bleu_score = self.calc_weighted_bleu(ref_tokens_list, hyp_tokens)
        
        # 3. Calculate AST match (use first reference for AST/dataflow as per paper)
        ast_match_score = self.calc_ast_match(references[0], hypothesis)
        
        # 4. Calculate data-flow match
        dataflow_match_score = self.calc_dataflow_match(references[0], hypothesis)
        
        # 5. Calculate final CodeBLEU
        codebleu = (self.alpha * bleu_score + 
                   self.beta * weighted_bleu_score + 
                   self.gamma * ast_match_score + 
                   self.delta * dataflow_match_score)
        
        return {
            'bleu': bleu_score,
            'weighted_bleu': weighted_bleu_score,
            'ast_match': ast_match_score,
            'dataflow_match': dataflow_match_score,
            'codebleu': codebleu
        }


# Convenience functions for easy importing

def detect_language(code: str) -> str:
    """
    Automatically detect whether code is Dart or Swift.
    
    Args:
        code: Source code string
        
    Returns:
        'dart' or 'swift'
        
    Raises:
        ValueError: If language cannot be determined
    """
    # Dart-specific patterns (weighted by specificity)
    dart_strong_indicators = [
        r'\bWidget\b', r'\bStatelessWidget\b', r'\bStatefulWidget\b',
        r'\bBuildContext\b', r'\bScaffold\b', r'\bFlutter\b',
        r'@override\s+Widget\s+build', r'pubspec\.yaml',
        r'\bmain\(\)\s*async\s*\{', r'\bFuture<\w+>',
        r'\bawait\s+\w+\.then\b', r'\bStreamController\b'
    ]
    
    dart_moderate_indicators = [
        r'\bvoid\s+main\(\)', r'\bvar\s+\w+\s*=', r'\bfinal\s+\w+\s*=',
        r'\bconst\s+\w+\s*=', r'\bclass\s+\w+\s+extends\s+',
        r'print\([\'"].*[\'"]\)', r'\blate\s+\w+', r'\bdynamic\s+\w+',
        r'=>\s*\{', r'\$\{?\w+\}?'  # String interpolation
    ]
    
    # Swift-specific patterns (weighted by specificity)
    swift_strong_indicators = [
        r'\bfunc\s+\w+\([^)]*\)\s*->\s*\w+', r'\bvar\s+\w+:\s*\w+',
        r'\blet\s+\w+:\s*\w+', r'@IBOutlet', r'@IBAction',
        r'@Published', r'@State', r'@ObservedObject', r'@EnvironmentObject',
        r'\bimport\s+UIKit\b', r'\bimport\s+SwiftUI\b', r'\bimport\s+Foundation\b',
        r'\bstruct\s+\w+\s*:\s*View\b', r'\.swift\b', r'\bself\.\w+',
        r'init\([^)]*\)\s*\{'
    ]
    
    swift_moderate_indicators = [
        r'\bfunc\s+\w+\(', r'\bvar\s+\w+\s*=', r'\blet\s+\w+\s*=',
        r'\bclass\s+\w+\s*:\s*\w+', r'\bstruct\s+\w+\s*\{',
        r'\benum\s+\w+\s*\{', r'\bprotocol\s+\w+\s*\{',
        r'\bguard\s+let\b', r'\bif\s+let\b', r'\\\\\([^)]+\)',  # String interpolation
        r'\bswitch\s+\w+\s*\{'
    ]
    
    import re
    
    # Calculate scores
    dart_score = 0
    swift_score = 0
    
    # Check strong indicators (worth 2 points each)
    for pattern in dart_strong_indicators:
        if re.search(pattern, code, re.MULTILINE):
            dart_score += 2
    
    for pattern in swift_strong_indicators:
        if re.search(pattern, code, re.MULTILINE):
            swift_score += 2
    
    # Check moderate indicators (worth 1 point each)
    for pattern in dart_moderate_indicators:
        if re.search(pattern, code, re.MULTILINE):
            dart_score += 1
    
    for pattern in swift_moderate_indicators:
        if re.search(pattern, code, re.MULTILINE):
            swift_score += 1
    
    # Additional checks for common keywords
    if 'flutter' in code.lower():
        dart_score += 3
    if 'uikit' in code.lower() or 'swiftui' in code.lower():
        swift_score += 3
    
    # Make decision
    if dart_score > swift_score:
        return 'dart'
    elif swift_score > dart_score:
        return 'swift'
    else:
        # Try to parse with each language as last resort
        try:
            lang_dart = Language(tree_sitter_dart.language())
            parser = Parser(lang_dart)
            tree = parser.parse(bytes(code, 'utf8'))
            if tree.root_node.has_error:
                raise Exception()
            return 'dart'
        except:
            try:
                lang_swift = Language(tree_sitter_swift.language())
                parser = Parser(lang_swift)
                tree = parser.parse(bytes(code, 'utf8'))
                if tree.root_node.has_error:
                    raise Exception()
                return 'swift'
            except:
                raise ValueError(
                    "Could not automatically detect language. "
                    "Please specify 'dart' or 'swift' explicitly."
                )


def codebleu_score(references: Union[str, List[str]], hypothesis: str, 
                   lang: Optional[str] = None,
                   weights: Tuple[float, float, float, float] = DEFAULT_WEIGHTS) -> float:
    """
    Calculate CodeBLEU score between reference(s) and hypothesis.
    
    Args:
        references: Reference code(s) - string or list of strings
        hypothesis: Generated code string
        lang: Programming language ('dart' or 'swift'). If None, auto-detect.
        weights: Weights (α, β, γ, δ) for (BLEU, weighted_BLEU, AST_match, dataflow_match)
                Default: (0.25, 0.25, 0.25, 0.25)
    
    Returns:
        CodeBLEU score between 0 and 1
    
    Example:
        >>> ref = "func hello() { print('Hello World') }"
        >>> hyp = "func hello() { print('Hello') }"
        >>> score = codebleu_score(ref, hyp, 'swift')
        >>> # Or with auto-detection:
        >>> score = codebleu_score(ref, hyp)
    """
    # Auto-detect language if not specified
    if lang is None:
        # Use first reference for detection
        ref_code = references if isinstance(references, str) else references[0]
        lang = detect_language(ref_code)
    
    calculator = CodeBLEUCalculator(lang, *weights)
    results = calculator.compute_codebleu(references, hypothesis)
    return results['codebleu']


def compute_codebleu(references: Union[str, List[str]], hypothesis: str, lang: str,
                    weights: Tuple[float, float, float, float] = DEFAULT_WEIGHTS,
                    return_all_scores: bool = False) -> Union[float, Dict[str, float]]:
    """
    Compute CodeBLEU with option to return all component scores.
    
    Args:
        references: Reference code(s)
        hypothesis: Generated code
        lang: Programming language ('dart' or 'swift')
        weights: Component weights (α, β, γ, δ)
        return_all_scores: If True, return dict with all scores
    
    Returns:
        CodeBLEU score or dict with all component scores
    """
    calculator = CodeBLEUCalculator(lang, *weights)
    results = calculator.compute_codebleu(references, hypothesis)
    
    if return_all_scores:
        return results
    return results['codebleu']


def corpus_codebleu(references_list: List[Union[str, List[str]]], 
                   hypotheses_list: List[str], lang: str,
                   weights: Tuple[float, float, float, float] = DEFAULT_WEIGHTS) -> Dict[str, float]:
    """
    Calculate CodeBLEU scores for a corpus (multiple examples).
    
    Args:
        references_list: List of references (each can be string or list of strings)
        hypotheses_list: List of hypotheses
        lang: Programming language
        weights: Component weights
    
    Returns:
        Dictionary with average scores and list of individual scores
    """
    if len(references_list) != len(hypotheses_list):
        raise ValueError("Number of references and hypotheses must match")
    
    calculator = CodeBLEUCalculator(lang, *weights)
    all_scores = []
    
    for refs, hyp in zip(references_list, hypotheses_list):
        scores = calculator.compute_codebleu(refs, hyp)
        all_scores.append(scores)
    
    # Calculate corpus-level averages
    avg_scores = {
        'bleu': np.mean([s['bleu'] for s in all_scores]),
        'weighted_bleu': np.mean([s['weighted_bleu'] for s in all_scores]),
        'ast_match': np.mean([s['ast_match'] for s in all_scores]),
        'dataflow_match': np.mean([s['dataflow_match'] for s in all_scores]),
        'codebleu': np.mean([s['codebleu'] for s in all_scores]),
        'individual_scores': all_scores
    }
    
    return avg_scores


# Example usage
if __name__ == "__main__":
    # Example 1: Dart code
    dart_reference = """
    void main() {
        var name = 'Alice';
        var age = 30;
        print('Name: $name, Age: $age');
        
        for (int i = 0; i < 5; i++) {
            print('Count: $i');
        }
    }
    """
    
    dart_hypothesis = """
    void main() {
        String name = 'Alice';
        int age = 30;
        print('Name: $name, Age: $age');
        
        for (var i = 0; i < 5; i++) {
            print('Number: $i');
        }
    }
    """
    
    # Example 2: Swift code with more keywords to show weighting effect
    swift_reference = """
    class Calculator {
        func calculateSum(numbers: [Int]) -> Int {
            var total = 0
            for num in numbers {
                total += num
            }
            return total
        }
    }
    
    let calc = Calculator()
    let result = calc.calculateSum(numbers: [1, 2, 3, 4, 5])
    print("Sum: \\(result)")
    """
    
    swift_hypothesis = """
    struct Calculator {
        func calculateSum(numbers: [Int]) -> Int {
            var sum = 0
            for number in numbers {
                sum = sum + number
            }
            return sum
        }
    }
    
    let calc = Calculator()
    let result = calc.calculateSum(numbers: [1, 2, 3, 4, 5])
    print("Sum: \\(result)")
    """
    
    # Test Dart CodeBLEU
    print("=== Dart Example ===")
    dart_calculator = CodeBLEUCalculator('dart')
    dart_scores = dart_calculator.compute_codebleu(dart_reference, dart_hypothesis)
    print(f"BLEU Score: {dart_scores['bleu']:.4f}")
    print(f"Weighted BLEU Score (keywords get {KEYWORD_WEIGHT}x weight): {dart_scores['weighted_bleu']:.4f}")
    print(f"AST Match Score (ignoring leaves): {dart_scores['ast_match']:.4f}")
    print(f"Data-flow Match Score: {dart_scores['dataflow_match']:.4f}")
    print(f"CodeBLEU Score: {dart_scores['codebleu']:.4f}")
    
    # Test Swift CodeBLEU
    print("\n=== Swift Example ===")
    swift_calculator = CodeBLEUCalculator('swift')
    swift_scores = swift_calculator.compute_codebleu(swift_reference, swift_hypothesis)
    print(f"BLEU Score: {swift_scores['bleu']:.4f}")
    print(f"Weighted BLEU Score (keywords get {KEYWORD_WEIGHT}x weight): {swift_scores['weighted_bleu']:.4f}")
    print(f"AST Match Score (ignoring leaves): {swift_scores['ast_match']:.4f}")
    print(f"Data-flow Match Score: {swift_scores['dataflow_match']:.4f}")
    print(f"CodeBLEU Score: {swift_scores['codebleu']:.4f}")
    
    # Show the effect of leaf node filtering
    print("\n=== AST Node Analysis (Swift) ===")
    swift_bytes = bytes(swift_reference, 'utf8')
    tree = swift_calculator.parser.parse(swift_bytes)
    
    all_nodes = swift_calculator.get_ast_subtrees(tree.root_node, swift_bytes, ignore_leaves=False)
    internal_nodes = swift_calculator.get_ast_subtrees(tree.root_node, swift_bytes, ignore_leaves=True)
    
    print(f"Total AST nodes (including leaves): {len(all_nodes)}")
    print(f"Internal AST nodes only: {len(internal_nodes)}")
    print(f"Leaf nodes filtered out: {len(all_nodes) - len(internal_nodes)}")
    
    # Test with multiple references
    print("\n=== Multiple References Example ===")
    multi_refs = [swift_reference, 
                  """struct Calculator {
                      func calculateSum(numbers: [Int]) -> Int {
                          return numbers.reduce(0, +)
                      }
                  }"""]
    multi_score = codebleu_score(multi_refs, swift_hypothesis, 'swift')
    print(f"CodeBLEU with multiple references: {multi_score:.4f}")