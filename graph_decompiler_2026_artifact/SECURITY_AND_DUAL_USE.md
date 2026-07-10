# Security and Dual-Use Statement

Neural decompilation can support authorized reverse engineering, malware analysis, vulnerability research, interoperability, incident response, and software maintenance. It can also reduce the cost of inspecting proprietary code or modifying malicious binaries.

This artifact releases only:

- public/generated benchmark source;
- derived benchmark assembly, CFG, and DFG records;
- model adapters and training/evaluation code;
- prediction and metric outputs.

It does not include proprietary application binaries, private source trees, credentials, exploit chains, or malware samples.

Users are responsible for obtaining authorization to analyze software and for complying with licenses and applicable law. Generated candidates are hypotheses, not trustworthy recovered source. The reported specialized pass@10 remains below 0.2, and every generated program should be independently compiled, tested, and reviewed before use in a security decision.

