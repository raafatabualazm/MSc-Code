@pragma('vm:entry-point')
bool hasBalancedDiceBonusRounds(List<String> events) {
  List<int> stack = [];
  for (String event in events) {
    if (event == 'B') {
      stack.add(0);
    } else if (event == 'C') {
      if (stack.isEmpty) return false;
      int total = stack.removeLast();
      if (total != 7 && total != 9) return false;
    } else if (event.length == 2 && event[0] == 'R') {
      if (stack.isEmpty) return false;
      int roll = int.parse(event[1]);
      int next = stack.removeLast() + roll;
      if (roll < 1 || roll > 6 || next > 9) return false;
      stack.add(next);
    } else {
      return false;
    }
  }
  return stack.isEmpty;
}

@pragma('vm:entry-point')
void main() {
  assert(hasBalancedDiceBonusRounds(['B', 'R3', 'R4', 'C']) == true);
  assert(hasBalancedDiceBonusRounds(['B', 'R6', 'R4', 'C']) == false);
  assert(hasBalancedDiceBonusRounds(['C']) == false);
  print('All tests passed!');
}