@pragma('vm:entry-point')
bool dnaBothInSameEpoch(String seq1, String seq2) {
  int epochIndex(String s) {
    if (s.isEmpty) return 0;
    int len = s.length;
    return (len + 3) ~/ 4;
  }
  return epochIndex(seq1) == epochIndex(seq2);
}

@pragma('vm:entry-point')
void main() {
  assert(dnaBothInSameEpoch('', '') == true);
  assert(dnaBothInSameEpoch('ACGT', 'ACGTA') == false);
  assert(dnaBothInSameEpoch('ACGTA', 'ACGTACGT') == true);
  print('All tests passed!');
}