@pragma('vm:entry-point')
List<String> rebuildTournamentAdvancers(List<String> events) {
  List<String> stack = [];
  List<String> sealed = [];
  for (String e in events) {
    if (e == '(') {
      stack.add('|');
    } else if (e == ')') {
      List<String> round = [];
      while (stack.isNotEmpty && stack.last != '|') {
        round.add(stack.removeLast());
      }
      if (stack.isNotEmpty) stack.removeLast();
      if (round.isNotEmpty) {
        round.sort();
        sealed.add(round.first);
        stack.add(round.first);
      }
    } else if (e == 'UNDO') {
      if (sealed.isNotEmpty && stack.isNotEmpty && stack.last == sealed.last) {
        stack.removeLast();
        sealed.removeLast();
      }
    } else {
      stack.add(e);
    }
  }
  return stack.where((s) => s != '|').toList();
}

@pragma('vm:entry-point')
void main() {
  assert(rebuildTournamentAdvancers([]).isEmpty);
  assert(rebuildTournamentAdvancers(['(', 'Owls', 'Bears', ')']).toString() == '[Bears]');
  assert(rebuildTournamentAdvancers(['(', 'Owls', 'Bears', ')', 'UNDO']).length == 0);
  print('All tests passed!');
}