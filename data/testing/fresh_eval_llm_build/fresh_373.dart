@pragma('vm:entry-point')
bool hasReplayPairInLogs(List<String> logs) {
  if (logs.length < 2) return false;
  List<int> ids = [];
  for (var line in logs) {
    ids.add(int.parse(line.split('#').last));
  }
  int left = 0;
  int right = ids.length - 1;
  while (left < right) {
    int mid = (left + right) ~/ 2;
    if (mid > 0 && ids[mid] == ids[mid - 1]) {
      return true;
    }
    if (ids[mid] - ids[0] >= mid) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left > 0 && ids[left] == ids[left - 1];
}

@pragma('vm:entry-point')
void main() {
  assert(hasReplayPairInLogs([]) == false);
  assert(hasReplayPairInLogs(['api#41', 'api#41', 'api#42']) == true);
  assert(hasReplayPairInLogs(['db#7', 'db#8', 'db#9']) == false);
  print('All tests passed!');
}