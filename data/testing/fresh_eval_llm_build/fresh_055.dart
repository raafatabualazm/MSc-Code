@pragma('vm:entry-point')
List<String> tokenizeDnaSignals(String strand, int minRun) {
  List<String> tokens = [];
  if (minRun < 1) return tokens;
  int i = 0;
  while (i < strand.length) {
    String ch = strand[i];
    bool valid = ch == 'A' || ch == 'C' || ch == 'G' || ch == 'T';
    if (!valid) {
      i++;
      continue;
    }
    int j = i;
    while (j < strand.length) {
      String c = strand[j];
      if (c != 'A' && c != 'C' && c != 'G' && c != 'T') break;
      j++;
    }
    int k = i;
    while (k < j) {
      int run = 1;
      while (k + run < j && strand[k + run] == strand[k]) {
        run++;
      }
      if (run >= minRun) {
        tokens.add('${strand[k]}$run');
      } else {
        int gc = 0;
        for (int m = k; m < k + run; m++) {
          if (strand[m] == 'G' || strand[m] == 'C') gc++;
        }
        if (gc == run && run > 1) {
          tokens.add('GC$run');
        } else if (run == 1 && k > i && k + 1 < j && strand[k - 1] == strand[k + 1]) {
          tokens.add('swap${strand[k]}');
        }
      }
      k += run;
    }
    i = j;
  }
  return tokens;
}

@pragma('vm:entry-point')
void main() {
  assert(tokenizeDnaSignals('AAA', 2).length == 1);
  assert(tokenizeDnaSignals('AGA', 2).toString() == '[swapG]');
  assert(tokenizeDnaSignals('AAAA', 0).length == 0);
  print('All tests passed!');
}