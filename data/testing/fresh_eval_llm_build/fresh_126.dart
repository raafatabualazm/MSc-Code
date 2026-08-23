@pragma('vm:entry-point')
String firstSlowServerLog(List<String> logs) {
  int lo = 0;
  int hi = logs.length;
  while (lo < hi) {
    int mid = (lo + hi) ~/ 2;
    int latency = int.parse(logs[mid].split('|')[0]);
    if (latency < 500) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo < logs.length ? logs[lo] : '';
}

@pragma('vm:entry-point')
void main() {
  assert(firstSlowServerLog([]) == '');
  assert(firstSlowServerLog(['120|boot ok', '500|db stall']) == '500|db stall');
  assert(firstSlowServerLog(['499|cache warm', '501|timeout']) == '501|timeout');
  print('All tests passed!');
}