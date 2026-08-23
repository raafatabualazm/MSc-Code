@pragma('vm:entry-point')
String? unpairedCore(String dna) {
  if (dna.isEmpty) return null;
  if (dna.length == 1) return dna;
  String first = dna[0];
  String last = dna[dna.length - 1];
  if ((first == 'A' && last == 'T') || (first == 'T' && last == 'A') ||
      (first == 'C' && last == 'G') || (first == 'G' && last == 'C')) {
    return unpairedCore(dna.substring(1, dna.length - 1));
  }
  return dna;
}

@pragma('vm:entry-point')
void main() {
  assert(unpairedCore('') == null);
  assert(unpairedCore('A') == 'A');
  assert(unpairedCore('AT') == null);
  print('All tests passed!');
}