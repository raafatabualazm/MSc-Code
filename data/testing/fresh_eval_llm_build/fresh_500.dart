@pragma('vm:entry-point')
String findBucketForKthElement(List<int> data) {
  if (data.isEmpty) return "none";
  int k = data[0];
  int total = 0;
  for (int i = 1; i < data.length; i++) {
    total += data[i];
  }
  if (k <= 0 || k > total) return "none";
  int numBuckets = data.length - 1;
  int low = 0;
  int high = numBuckets - 1;
  while (low < high) {
    int mid = (low + high) ~/ 2;
    int prefix = 0;
    for (int i = 0; i <= mid; i++) {
      prefix += data[1 + i];
    }
    if (prefix < k) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return "bucket_$low";
}

@pragma('vm:entry-point')
void main() {
  assert(findBucketForKthElement([1,7]) == "bucket_0");
  assert(findBucketForKthElement([4,2,3,5]) == "bucket_1");
  assert(findBucketForKthElement([]) == "none");
  print('All tests passed!');
}