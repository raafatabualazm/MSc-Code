@pragma('vm:entry-point')
String reportBalancedDiceWindow(List<int> rounds) {
  if (rounds.isEmpty) return 'none';
  List<int> freq = List.filled(7, 0);
  int left = 0, high = 0, low = 0, sum = 0;
  int bestLen = 0, bestStart = -1, bestSum = -1;
  for (int right = 0; right < rounds.length; right++) {
    int v = rounds[right];
    if (v < 1 || v > 6) {
      while (left <= right) {
        int gone = rounds[left++];
        if (gone >= 1 && gone <= 6) {
          freq[gone]--;
          sum -= gone;
          gone >= 4 ? high-- : low--;
        }
      }
      continue;
    }
    freq[v]++;
    sum += v;
    v >= 4 ? high++ : low++;
    while (freq[v] > 2 || (high - low).abs() > 1) {
      int gone = rounds[left++];
      freq[gone]--;
      sum -= gone;
      gone >= 4 ? high-- : low--;
    }
    int len = right - left + 1;
    if (len > bestLen || (len == bestLen && sum > bestSum)) {
      bestLen = len;
      bestStart = left;
      bestSum = sum;
    }
  }
  return bestLen == 0 ? 'none' : '$bestStart-${bestStart + bestLen - 1}:$bestSum';
}

@pragma('vm:entry-point')
void main() {
  assert(reportBalancedDiceWindow([]) == 'none');
  assert(reportBalancedDiceWindow([6, 1]) == '0-1:7');
  assert(reportBalancedDiceWindow([4, 1, 4, 1, 4]) == '0-3:10');
  print('All tests passed!');
}