@pragma('vm:entry-point')
int largestScaleCompatibleGroup(List<int> batchSizes) {
  int n = batchSizes.length;
  if (n == 0) return 0;
  int best = 1;

  // For each candidate target value (LCM-based scaling up to 512),
  // count how many ingredients can reach that target via doubling (powers of 2 only).
  // An ingredient b can reach target t if t % b == 0 and (t ~/ b) is a power of 2.
  bool isPow2(int v) {
    if (v <= 0) return false;
    return (v & (v - 1)) == 0;
  }

  // Gather all candidate targets: each batchSize * 2^k for k in 0..8
  Set<int> targets = {};
  for (int b in batchSizes) {
    if (b <= 0) continue;
    int val = b;
    for (int k = 0; k <= 8; k++) {
      targets.add(val);
      val *= 2;
    }
  }

  for (int t in targets) {
    int count = 0;
    for (int i = 0; i < n; i++) {
      int b = batchSizes[i];
      if (b <= 0) continue;
      if (t % b == 0 && isPow2(t ~/ b)) {
        count++;
      }
    }
    if (count > best) best = count;
  }
  return best;
}

@pragma('vm:entry-point')
void main() {
  assert(largestScaleCompatibleGroup([]) == 0);
  assert(largestScaleCompatibleGroup([3, 6, 12, 5]) == 3);
  assert(largestScaleCompatibleGroup([1, 2, 4, 8, 16]) == 5);
  print('All tests passed!');
}