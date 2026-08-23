@pragma('vm:entry-point')
int dnaBaseIntervalStreak(String dna) {
  const baseIndex = {'A': 0, 'C': 1, 'G': 2, 'T': 3};
  if (dna.length < 2) return 0;
  int total = 0;
  int streak = 0;
  for (int i = 0; i < dna.length - 1; i++) {
    final int? curIdx = baseIndex[dna[i]];
    final int? nxtIdx = baseIndex[dna[i + 1]];
    if (curIdx == null || nxtIdx == null) {
      streak = 0;
      continue;
    }
    final int diff = nxtIdx - curIdx;
    if (diff == 0) {
      streak = 0;
      continue;
    } else if (diff > 0) {
      streak += 1;
      total += diff * streak;
    } else {
      streak = 0;
      total += -diff;
    }
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(dnaBaseIntervalStreak('ACGT') == 6);
  assert(dnaBaseIntervalStreak('') == 0);
  assert(dnaBaseIntervalStreak('ACGTA') == 9);
  print('All tests passed!');
}