@pragma('vm:entry-point')
String describeHashBucketLanding(List<int> bucketCounts, int probe) {
  if (bucketCounts.isEmpty) return 'no-buckets';
  int low = 0;
  int high = bucketCounts.length;
  while (low < high) {
    int mid = (low + high) >> 1;
    if (bucketCounts[mid] < probe) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  if (low == bucketCounts.length) return 'after-max';
  if (bucketCounts[low] == probe) {
    int end = low;
    while (end + 1 < bucketCounts.length && bucketCounts[end + 1] == probe) {
      end++;
    }
    return end == low ? 'exact:$low' : 'plateau:$low-$end';
  }
  return low == 0 ? 'before-min' : 'gap:${bucketCounts[low - 1]}-${bucketCounts[low]}';
}

@pragma('vm:entry-point')
void main() {
  assert(describeHashBucketLanding([], 4) == 'no-buckets');
  assert(describeHashBucketLanding([2, 5, 5, 8], 5) == 'plateau:1-2');
  assert(describeHashBucketLanding([1, 4, 9], 6) == 'gap:4-9');
  print('All tests passed!');
}