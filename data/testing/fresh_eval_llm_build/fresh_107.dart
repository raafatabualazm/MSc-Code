@pragma('vm:entry-point')
List<List<int>> traceBucketCollisionWindows(List<int> bucketCounts, int collisionLimit) {
  List<List<int>> windows = [];
  int left = 0;
  int sum = 0;
  for (int right = 0; right < bucketCounts.length; right++) {
    sum += bucketCounts[right];
    if (sum >= collisionLimit) {
      windows.add([left, right]);
      sum -= bucketCounts[left++];
    }
  }
  return windows;
}

@pragma('vm:entry-point')
void main() {
  assert(traceBucketCollisionWindows([], 3).toString() == '[]');
  assert(traceBucketCollisionWindows([1, 1, 1], 2).toString() == '[[0, 1], [1, 2]]');
  assert(traceBucketCollisionWindows([4, 0, 2], 4).toString() == '[[0, 0]]');
  print('All tests passed!');
}