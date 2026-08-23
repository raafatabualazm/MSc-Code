import pathlib
import hashlib
import json
import sys
import tempfile
import unittest


ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import build_scrubbed_dataset as builder
import audit_scrubbed_release as release_audit


class ScrubbedAssemblySafetyTests(unittest.TestCase):
    def setUp(self):
        source='''
class TreeNode {
  int add(int a, int b) => a + b;
}
@pragma('vm:entry-point')
void c_0123456789ab() {}
'''
        self.symbols=builder.source_symbol_names(source)

    def test_source_symbol_inventory_includes_functions_and_types(self):
        self.assertIn('TreeNode',self.symbols['types'])
        self.assertIn('add',self.symbols['functions'])

    def test_helper_named_like_opcode_never_rewrites_instruction(self):
        actual=builder.scrub_objdump_instruction(
            'add    rax,rax', 'c_0123456789ab', self.symbols,
        )
        self.assertEqual(actual,'add    rax,rax')

    def test_names_are_neutralized_only_inside_symbol_annotation(self):
        constructor=builder.scrub_objdump_instruction(
            'call   99a20 <new TreeNode>', 'c_0123456789ab', self.symbols,
        )
        method=builder.scrub_objdump_instruction(
            'call   99a30 <TreeNode.add>', 'c_0123456789ab', self.symbols,
        )
        recursive=builder.scrub_objdump_instruction(
            'jmp    99a40 <c_0123456789ab+0x10>', 'c_0123456789ab', self.symbols,
        )
        self.assertEqual(constructor,'call   0x99a20 <new type_0>')
        self.assertEqual(method,'call   0x99a30 <type_0.local_0>')
        self.assertEqual(recursive,'jmp    0x99a40 <candidate+0x10>')

    def test_formatter_has_no_file_id_or_synthetic_signature(self):
        assembly=builder.format_scrubbed_assembly(
            [(0x1000,'push rbp'),(0x1001,'add rax,rax'),(0x1004,'ret')],
            'c_0123456789ab',self.symbols,
        )
        self.assertNotIn('File file:///',assembly)
        self.assertNotIn('static void candidate(void)',assembly)
        self.assertNotIn('TreeNode',assembly)
        self.assertIn('\tadd rax,rax',assembly)

    def test_unknown_or_corrupted_mnemonic_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'unknown_or_corrupted_mnemonic'):
            builder.validate_instruction_mnemonic('local_0 rax,rax')

    def test_symbol_residue_gate_detects_unscrubbed_type(self):
        leaked='0x1000 <+0>:\tcall 0x2000 <new TreeNode>\n'
        self.assertEqual(builder.symbol_residue(leaked,self.symbols),['TreeNode'])

    def test_model_row_is_a_strict_allowlist(self):
        common={key:None for key in builder.PUBLIC_MODEL_FIELDS}
        common.update({
            'task_id':'source-derived-id',
            'filename':'source-derived-id.dart',
            'benchmark_protocol':{'original_source_sha256':'secret'},
            'graph_v2':{'assembly_sha256':'secret'},
        })
        row=builder.model_facing_row(common)
        self.assertEqual(tuple(row),builder.PUBLIC_MODEL_FIELDS)
        self.assertNotIn('task_id',row)
        self.assertNotIn('filename',row)
        self.assertNotIn('benchmark_protocol',row)
        self.assertNotIn('graph_v2',row)

    def test_independent_auditor_accepts_only_the_allowlisted_schema(self):
        source="@pragma('vm:entry-point')\nvoid candidate() {}\n"
        assembly=builder.format_scrubbed_assembly(
            [(0x1000,'push rbp'),(0x1001,'pop rbp'),(0x1002,'ret')],
            'candidate',builder.source_symbol_names(source),
        )
        block={'id':0,'instructions':['push rbp','pop rbp','ret']}
        public=builder.model_facing_row({
            'lang':'Dart','function':'candidate','camel_case_function_name':'candidate',
            'python_function_name':'','dart_function_signature':'',
            'prompt_signature_mode':'name_only','assembly':assembly,'cfg':[block],
            'edges':[],
        })
        private={'dart_source':source,**public,
                 'evaluation_only_dart_function_signature':'void candidate()',
                 'tests':'void main() { candidate(); }'}
        ledger={
            'assembly_sha256':hashlib.sha256(assembly.encode()).hexdigest(),
            'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'canonical_source_sha256':'source-1','canonical_assembly_sha256':'assembly-1',
            'neutral_id':'ledger-only-id','main_transformed':False,
            'test_kind':'dart_harness',
        }
        with tempfile.TemporaryDirectory() as temp:
            directory=pathlib.Path(temp)
            rows={
                'master_dart_graphv2_signature_scrubbed_public.jsonl':[public],
                'master_dart_graphv2_signature_scrubbed_private.jsonl':[private],
                'master_dart_graphv2_compile_ledger.jsonl':[ledger],
                'master_dart_graphv2_quarantine.jsonl':[],
            }
            for name,items in rows.items():
                (directory/name).write_text(''.join(json.dumps(item)+'\n' for item in items))
            result=release_audit.audit(directory,pathlib.Path('missing-dart'),pathlib.Path('missing-aot'),skip_runtime=True)
        self.assertTrue(result['passed'],result['problems'])


if __name__=='__main__':
    unittest.main()
