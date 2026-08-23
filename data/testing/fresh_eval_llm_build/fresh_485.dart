@pragma('vm:entry-point')
int countOverloadedHashBuckets(List<int> keys) {
  if (keys.isEmpty) return 0;
  int bucketCount = keys.length <= 8 ? 4 : 7;
  List<int> buckets = List.filled(bucketCount, 0);
  for (int key in keys) {
    int k = key < 0 ? -key : key;
    int idx = k % bucketCount;
    buckets[idx]++;
  }
  int total = keys.length;
  int overloaded = 0;
  for (int count in buckets) {
    if (count * bucketCount > total) {
      overloaded++;
    }
  }
  return overloaded;
}

@pragma('vm:entry-point')
void main() {
  assert(countOverloadedHashBuckets([]) == 0);
  assert(countOverloadedHashBuckets([5]) == 1);
  assert(countOverloadedHashBuckets([0,1,2,3,4,5,6,7]) == 0);
  print('All tests passed!');
}