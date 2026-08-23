@pragma('vm:entry-point')
int accumulatedRepeatGenePower(String dna) {
  int run = 0;
  int total = 0;
  for (int i = 0; i < dna.length; i++) {
    run = (i > 0 && dna[i] == dna[i - 1]) ? run + 1 : 1;
    total += run;
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(accumulatedRepeatGenePower('') == 0);
  assert(accumulatedRepeatGenePower('AA') == 3);
  assert(accumulatedRepeatGenePower('ATTTGC') == 9);
  print('All tests passed!');
}