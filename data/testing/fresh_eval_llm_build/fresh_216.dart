@pragma('vm:entry-point')
int barcodeDigitDriftScore(List<int> scans) {
  if (scans.isEmpty) return 0;
  final freq = <int, int>{};
  for (final v in scans) {
    if (v < 0 || v > 9) continue;
    freq[v] = (freq[v] ?? 0) + 1;
  }
  if (freq.isEmpty) return -1;
  final digits = freq.keys.toList()
    ..sort((a, b) {
      final byFreq = freq[b]!.compareTo(freq[a]!);
      if (byFreq != 0) return byFreq;
      if (a.isOdd != b.isOdd) return a.isOdd ? -1 : 1;
      return a.compareTo(b);
    });
  var score = 0;
  for (var i = 0; i < digits.length; i++) {
    final d = digits[i];
    var run = 0;
    for (final v in scans) {
      if (v != d) {
        if (run >= 2) score += run * (i + 1);
        run = 0;
        continue;
      }
      run++;
      if (run == 3) score += d;
    }
    if (run >= 2) score += run * (i + 1);
    if (i > 0 && (d - digits[i - 1]).abs() == 1) {
      score -= 2;
    } else if (freq[d] == 1) {
      score += 1;
    }
  }
  return score + digits.length;
}

@pragma('vm:entry-point')
void main() {
  assert(barcodeDigitDriftScore([]) == 0);
  assert(barcodeDigitDriftScore([1, 1, 1]) == 5);
  assert(barcodeDigitDriftScore([9, 0, 9, 0, 9]) == 2);
  print('All tests passed!');
}