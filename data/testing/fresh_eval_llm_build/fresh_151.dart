@pragma('vm:entry-point')
int retainedBatteryCycles(String log, int reserve) {
  var stack = <int>[];
  for (var ch in log.split('')) {
    if (ch == 'C') {
      stack.add(2);
    } else if (ch == 'R' && stack.isNotEmpty) {
      stack.removeLast();
    }
  }
  return reserve + stack.fold(0, (a, b) => a + b);
}

@pragma('vm:entry-point')
void main() {
  assert(retainedBatteryCycles('', 5) == 5);
  assert(retainedBatteryCycles('CCR', 0) == 2);
  assert(retainedBatteryCycles('CRC', 1) == 3);
  print('All tests passed!');
}