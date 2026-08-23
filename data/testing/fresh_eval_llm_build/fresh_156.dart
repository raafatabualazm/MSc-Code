@pragma('vm:entry-point')
int countPositiveHeightsWithEvenFrequency(List<int> tideHeights) {
  var freq = <int, int>{};
  for (var h in tideHeights) {
    freq[h] = (freq[h] ?? 0) + 1;
  }
  return freq.keys.where((h) => h > 0 && freq[h]! % 2 == 0).length;
}

@pragma('vm:entry-point')
void main() {
  assert(countPositiveHeightsWithEvenFrequency([2, 2, -2, 3, 3, 3]) == 1);
  assert(countPositiveHeightsWithEvenFrequency([]) == 0);
  assert(countPositiveHeightsWithEvenFrequency([1, 1, 2, 2, 3]) == 2);
  print('All tests passed!');
}