@pragma('vm:entry-point')
bool hasGradualWeightIncrease(List<int> weights) {
  final n = weights.length;
  if (n < 2) return false;
  for (int start = 0; start < n - 1; start++) {
    for (int end = start + 1; end < n; end++) {
      for (int split = start; split < end; split++) {
        int sum1 = 0, sum2 = 0;
        int max1 = weights[start];
        int min2 = weights[split + 1];
        for (int i = start; i <= split; i++) {
          sum1 += weights[i];
          if (weights[i] > max1) max1 = weights[i];
        }
        for (int i = split + 1; i <= end; i++) {
          sum2 += weights[i];
          if (weights[i] < min2) min2 = weights[i];
        }
        if (sum1 < sum2 && max1 < min2) {
          return true;
        }
      }
    }
  }
  return false;
}

@pragma('vm:entry-point')
void main() {
  assert(hasGradualWeightIncrease([]) == false);
  assert(hasGradualWeightIncrease([1, 2]) == true);
  assert(hasGradualWeightIncrease([3, 2, 1]) == false);
  print('All tests passed!');
}