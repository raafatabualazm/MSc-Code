@pragma('vm:entry-point')
List<int> reconcileBucketQueueCounts(String events, int bucketCount) {
  List<int> counts = List.filled(bucketCount, 0);
  List<int> active = [];
  for (int i = 0; i < events.length; i++) {
    String c = events[i];
    if (c == '!') {
      if (active.isNotEmpty) {
        int b = active.removeLast();
        counts[b]--;
      }
    } else if (c == '-') {
      if (active.isNotEmpty) {
        int b = active.removeAt(0);
        counts[b]--;
      }
    } else {
      int b = c.codeUnitAt(0) % bucketCount;
      counts[b]++;
      if (counts[b] <= 2) {
        active.add(b);
      } else {
        counts[b]--;
      }
    }
  }
  return counts;
}

@pragma('vm:entry-point')
void main() {
  assert(reconcileBucketQueueCounts("", 3).toString() == "[0, 0, 0]");
  assert(reconcileBucketQueueCounts("AB-", 3).toString() == "[1, 0, 0]");
  assert(reconcileBucketQueueCounts("BBBB", 3).toString() == "[2, 0, 0]");
  print('All tests passed!');
}