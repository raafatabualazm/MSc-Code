@pragma('vm:entry-point')
List<int> trafficLightPhaseSequence(List<String> c) {
  List<int> stack = [];
  for (var cmd in c) {
    if (cmd == 'R') stack.add(1);
    else if (cmd == 'G') stack.add(2);
    else if (cmd == 'Y') stack.add(3);
    else if (cmd == 'UNDO') {
      if (stack.isNotEmpty) stack.removeLast();
    } else if (cmd == 'RESET') stack.clear();
  }
  return stack;
}

@pragma('vm:entry-point')
void main() {
  assert(trafficLightPhaseSequence([]).toString() == '[]');
  assert(trafficLightPhaseSequence(['RESET','G','Y']).toString() == '[2, 3]');
  assert(trafficLightPhaseSequence(['UNDO','R','UNDO']).toString() == '[]');
  print('All tests passed!');
}