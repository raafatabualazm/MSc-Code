@pragma('vm:entry-point')
bool isPacketHistogramPalindrome(List<int> packetSizes) {
  var freq = <int, int>{};
  for (var size in packetSizes) {
    freq[size] = (freq[size] ?? 0) + 1;
  }
  if (freq.length <= 1) return true;
  var sortedSizes = freq.keys.toList()..sort();
  List<int> counts = sortedSizes.map((s) => freq[s]!).toList();
  int n = counts.length;
  for (int i = 0; i < n ~/ 2; i++) {
    if (counts[i] != counts[n - 1 - i]) return false;
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(isPacketHistogramPalindrome([]) == true);
  assert(isPacketHistogramPalindrome([1,1,2,2]) == true);
  assert(isPacketHistogramPalindrome([1,2,1]) == false);
  print('All tests passed!');
}