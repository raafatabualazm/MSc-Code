@pragma('vm:entry-point')
int hashBucketUndoCount(String events, int initialBuckets) {
  var history = <int>[initialBuckets];
  for (var c in events.split('')) {
    if (c == '!') {
      if (history.length > 1) history.removeLast();
    } else {
      history.add(history.last + (c == '+' ? 1 : -1));
    }
  }
  return history.last;
}

@pragma('vm:entry-point')
void main() {
  assert(hashBucketUndoCount("+-!", 2) == 3);
  assert(hashBucketUndoCount("++--!!", 0) == 2);
  assert(hashBucketUndoCount("", 5) == 5);
  print('All tests passed!');
}