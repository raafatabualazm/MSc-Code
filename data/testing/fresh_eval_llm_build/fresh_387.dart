@pragma('vm:entry-point')
int trafficPhaseUndoScore(List<String> events) {
  List<String> stack = [];
  for (final e in events) {
    if (e == 'UNDO') {
      if (stack.isNotEmpty) stack.removeLast();
    } else if (e == 'FLASH') {
      if (stack.isNotEmpty) stack.add(stack.last);
    } else if (e == 'CLEAR') {
      stack.clear();
    } else {
      stack.add(e);
    }
  }
  int score = stack.length;
  for (int i = 1; i < stack.length; i++) {
    String a = stack[i - 1], b = stack[i];
    if ((a == 'G' && b == 'Y') || (a == 'Y' && b == 'R') || (a == 'R' && b == 'G')) {
      score += 2;
    } else if (a == b) {
      score -= 1;
    } else {
      score += 1;
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(trafficPhaseUndoScore([]) == 0);
  assert(trafficPhaseUndoScore(['G', 'Y', 'R']) == 7);
  assert(trafficPhaseUndoScore(['G', 'FLASH']) == 1);
  print('All tests passed!');
}