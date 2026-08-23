@pragma('vm:entry-point')
int recountCipherTally(String ledger) {
  if (ledger.isEmpty) return 0;
  int score = 0;
  int i = 0;
  String previous = '';
  while (i < ledger.length) {
    String candidate = ledger[i];
    int code = candidate.codeUnitAt(0);
    if (code < 65 || code > 90) return -1;
    i++;
    if (i >= ledger.length || ledger.codeUnitAt(i) < 48 || ledger.codeUnitAt(i) > 57) return -1;
    int count = 0;
    while (i < ledger.length && ledger.codeUnitAt(i) >= 48 && ledger.codeUnitAt(i) <= 57) {
      count = count * 10 + (ledger.codeUnitAt(i) - 48);
      i++;
    }
    if (count == 0) {
      previous = candidate;
    } else {
      for (int j = 1; j <= count; j++) {
        bool vowel = 'AEIOU'.contains(candidate);
        if (vowel) {
          score += j.isOdd ? 2 : 1;
        } else if (j.isOdd) {
          score += 1;
        } else {
          score -= 1;
        }
      }
      if (previous == candidate) score -= count;
      if (count % 5 == 0) score += 3;
      previous = candidate;
    }
    if (i < ledger.length) {
      if (ledger[i] != ',') return -1;
      i++;
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(recountCipherTally('A1') == 2);
  assert(recountCipherTally('B5,B1') == 4);
  assert(recountCipherTally('A0,B1') == 1);
  print('All tests passed!');
}