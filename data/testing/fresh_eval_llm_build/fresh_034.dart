@pragma('vm:entry-point')
int countMirroredTimeoutFrames(List<String> logs) {
  if (logs.length < 2) return 0;
  int frame = (logs.first.contains('TIMEOUT') &&
          logs.last.contains('TIMEOUT') &&
          logs.first.length == logs.last.length)
      ? 1
      : 0;
  return frame + countMirroredTimeoutFrames(logs.sublist(1, logs.length - 1));
}

@pragma('vm:entry-point')
void main() {
  assert(countMirroredTimeoutFrames([]) == 0);
  assert(countMirroredTimeoutFrames(['TIMEOUT-A', 'TIMEOUT-B']) == 1);
  assert(countMirroredTimeoutFrames(['TIMEOUT-AA', 'ok', 'TIMEOUT-B']) == 0);
  print('All tests passed!');
}