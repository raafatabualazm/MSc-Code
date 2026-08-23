@pragma('vm:entry-point')
bool canBalanceShelfLoadsByMinUnit(List<int> weights) {
  if (weights.length < 2) return false;
  int minVal = weights[0];
  for (int i = 1; i < weights.length; i++) {
    if (weights[i] < minVal) minVal = weights[i];
  }
  if (minVal <= 0) return false;
  // Check if any contiguous subarray of length>=2 has sum divisible by minVal
  // Use prefix sums mod minVal; if two equal mod values exist with index diff>=2
  List<int> prefixMod = List.filled(weights.length + 1, 0);
  for (int i = 0; i < weights.length; i++) {
    prefixMod[i + 1] = (prefixMod[i] + weights[i]) % minVal;
  }
  Map<int, int> firstSeen = {};
  for (int i = 0; i <= weights.length; i++) {
    int mod = prefixMod[i];
    if (firstSeen.containsKey(mod)) {
      if (i - firstSeen[mod]! >= 2) return true;
    } else {
      firstSeen[mod] = i;
    }
  }
  return false;
}

@pragma('vm:entry-point')
void main() {
  assert(canBalanceShelfLoadsByMinUnit([3, 6, 9]) == true);
  assert(canBalanceShelfLoadsByMinUnit([5, 7, 11]) == false);
  assert(canBalanceShelfLoadsByMinUnit([]) == false);
  print('All tests passed!');
}