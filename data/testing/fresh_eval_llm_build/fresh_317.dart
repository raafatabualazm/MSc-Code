@pragma('vm:entry-point')
int findSessionWrapInLogs(List<String> logs) {
  if (logs.isEmpty) return -1;
  int first = int.parse(logs.first.split(' ')[0]);
  int last = int.parse(logs.last.split(' ')[0]);
  if (first <= last) return -1;
  int low = 0;
  int high = logs.length - 1;
  while (low < high) {
    int mid = (low + high) ~/ 2;
    int value = int.parse(logs[mid].split(' ')[0]);
    if (value >= first) {
      int nextValue = int.parse(logs[mid + 1].split(' ')[0]);
      if (nextValue < value) return mid + 1;
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low;
}

@pragma('vm:entry-point')
void main() {
  assert(findSessionWrapInLogs([]) == -1);
  assert(findSessionWrapInLogs(['40 api-start','50 cache-hit','5 reboot']) == 2);
  assert(findSessionWrapInLogs(['7 steady','8 steady']) == -1);
  print('All tests passed!');
}