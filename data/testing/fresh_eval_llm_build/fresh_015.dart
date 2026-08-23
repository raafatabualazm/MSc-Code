@pragma('vm:entry-point')
List<int> qrModuleRunStates(String modules) {
  var states = <int>[];
  var run = 0;
  for (var ch in modules.split('')) {
    if (ch == 'X') {
      run++;
    } else if (run > 0) {
      states.add(run >= 3 ? 1 : -1);
      run = 0;
    }
  }
  if (run > 0) states.add(run >= 3 ? 1 : -1);
  return states;
}

@pragma('vm:entry-point')
void main() {
  assert(qrModuleRunStates('').isEmpty);
  assert(qrModuleRunStates('XXX.').toString() == '[1]');
  assert(qrModuleRunStates('XX.XXXX.X').length == 3);
  print('All tests passed!');
}