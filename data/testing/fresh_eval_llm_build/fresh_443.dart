@pragma('vm:entry-point')
List<int> locateKthInBucketCounts(List<int> bucketCounts, int k) {
  if (k < 0) return [];
  for (int i = 0; i < bucketCounts.length; i++) {
    int count = bucketCounts[i];
    if (k < count) {
      return [i, k];
    }
    k -= count;
  }
  return [];
}

@pragma('vm:entry-point')
void main() {
  assert(locateKthInBucketCounts([], 0).toString() == '[]');
  assert(locateKthInBucketCounts([5], 2).toString() == '[0, 2]');
  assert(locateKthInBucketCounts([2, 3, 1], 5).toString() == '[2, 0]');
  print('All tests passed!');
}