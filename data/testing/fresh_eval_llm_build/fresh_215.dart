@pragma('vm:entry-point')
List<double> hashBucketStableLengths(List<int> bucketCounts) {
  final int n = bucketCounts.length;
  if (n == 0) return [];
  if (n == 1) return [1.0];
  int minVal = bucketCounts[0], maxVal = bucketCounts[0];
  for (int i = 1; i < n; i++) {
    final v = bucketCounts[i];
    if (v < minVal) minVal = v;
    if (v > maxVal) maxVal = v;
  }
  final double threshold = (maxVal - minVal) / 2.0;
  final List<double> result = List<double>.filled(n, 0.0);
  for (int i = 0; i < n; i++) {
    if (threshold == 0.0) {
      result[i] = (n - i).toDouble();
      continue;
    }
    double total = 0.0;
    int j = i;
    while (j < n - 1) {
      final double diff = (bucketCounts[j + 1] - bucketCounts[j]).abs().toDouble();
      if (total + diff > threshold) break;
      total += diff;
      j++;
    }
    result[i] = (j - i + 1).toDouble();
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(hashBucketStableLengths([1, 2, 3]).toString() == '[2.0, 2.0, 1.0]');
  assert(hashBucketStableLengths([]).toString() == '[]');
  assert(hashBucketStableLengths([5]).toString() == '[1.0]');
  print('All tests passed!');
}