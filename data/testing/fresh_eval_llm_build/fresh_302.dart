@pragma('vm:entry-point')
String findQuantileBucketIndex(List<int> bucketCounts, String quantile) {
  int total = bucketCounts.fold(0, (a, b) => a + b);
  if (total == 0) return '-1';
  int target;
  if (quantile == 'min') {
    target = 1;
  } else if (quantile == 'q1') {
    target = (total + 3) ~/ 4;
  } else if (quantile == 'median') {
    target = (total + 1) ~/ 2;
  } else if (quantile == 'q3') {
    target = (3 * total + 3) ~/ 4;
  } else if (quantile == 'max') {
    target = total;
  } else {
    return '-1';
  }
  int cumulative = 0;
  for (int i = 0; i < bucketCounts.length; i++) {
    cumulative += bucketCounts[i];
    if (cumulative >= target) return i.toString();
  }
  return '-1';
}

@pragma('vm:entry-point')
void main() {
  assert(findQuantileBucketIndex([1, 2, 3], 'median') == '1');
  assert(findQuantileBucketIndex([], 'min') == '-1');
  assert(findQuantileBucketIndex([5], 'q3') == '0');
  print('All tests passed!');
}