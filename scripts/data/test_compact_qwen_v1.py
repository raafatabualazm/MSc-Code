import importlib.util
from pathlib import Path

P=Path(__file__).with_name("build_compact_qwen_v1.py")
S=importlib.util.spec_from_file_location("codec",P); M=importlib.util.module_from_spec(S); S.loader.exec_module(M)

def test_fake_signature_and_symbols_are_scrubbed():
    row={"cfg":[{"id":0,"start_address":"0x10","instructions":[M.FAKE_SIGNATURE,"call 0x20 <new SecretClass>","ret"]}],
         "edges":[],"integrity":{"entry_blocks":[0]}}
    c=M.canonicalize(row)
    assert c["blocks"][0]["instructions"]==["call @U0","ret"]
    assert "SecretClass" not in str(c)

def test_runtime_stub_semantics_survive_but_self_is_neutral():
    row={"cfg":[{"id":0,"start_address":"0x10","instructions":["call 0x20 <stub _iso_stub_AllocateArrayStub>","call 0x30 <candidate.<anonymous closure>>","ret"]}],"edges":[]}
    c=M.canonicalize(row)
    assert c["blocks"][0]["instructions"]==["call @STUB:AllocateArrayStub","call @SELF_CLOSURE","ret"]

def test_corrupt_local_opcode_is_rejected():
    row={"cfg":[{"id":0,"start_address":"0x10","instructions":["local_0 rax,rax"]}],"edges":[]}
    try:M.canonicalize(row)
    except ValueError as e:assert "unknown_or_corrupt_mnemonic" in str(e)
    else:raise AssertionError("corrupt opcode accepted")

def test_text_codec_roundtrip():
    c={"architecture":"x86_64","entry_blocks":[0],"blocks":[{"id":0,"instructions":["mov rax,rbx","ret"]}],
       "cfg_edges":[],"dfg_edges":[]}
    ex=["mov rax,rbx"]; text=M.encode(c,{ex[0]:0}); got=M.decode(text,ex); got["dfg_edges"]=[]
    assert got==c
