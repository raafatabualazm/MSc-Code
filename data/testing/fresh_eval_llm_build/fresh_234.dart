@pragma('vm:entry-point')
String runLengthComplementDNA(String dna) {
  if (dna.isEmpty) return '';
  const comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'};
  final buf = StringBuffer();
  int i = 0;
  while (i < dna.length) {
    final base = dna[i];
    int count = 1;
    while (i + count < dna.length && dna[i + count] == base) count++;
    buf.write(comp[base]!);
    if (count > 1) buf.write(count);
    i += count;
  }
  return buf.toString();
}

@pragma('vm:entry-point')
void main() {
  assert(runLengthComplementDNA('AATGC') == 'T2ACG');
  assert(runLengthComplementDNA('') == '');
  assert(runLengthComplementDNA('CCCGGG') == 'G3C3');
  print('All tests passed!');
}