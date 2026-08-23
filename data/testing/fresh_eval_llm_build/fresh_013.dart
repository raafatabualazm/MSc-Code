@pragma('vm:entry-point')
String? smallestBucketCountForLoadFactor(List<int> itemCounts) {
  if (itemCounts.isEmpty) return null;
  int maxItems = itemCounts.reduce((a, b) => a > b ? a : b);
  int minBuckets = ((maxItems * 4) + 2) ~/ 3; // ceil(maxItems / 0.75)
  if (minBuckets <= 1) return '1';
  int buckets = 1;
  while (buckets < minBuckets) {
    buckets <<= 1;
  }
  return buckets.toString();
}

@pragma('vm:entry-point')
void main() {
  assert(smallestBucketCountForLoadFactor([]) == null);
  assert(smallestBucketCountForLoadFactor([3]) == '4');
  assert(smallestBucketCountForLoadFactor([13]) == '32');
  print('All tests passed!');
}