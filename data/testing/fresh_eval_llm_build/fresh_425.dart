@pragma('vm:entry-point')
String simulateTournament(String input) {
  var parts = input.split(':');
  var bracket = parts[0];
  var results = parts.length > 1 ? parts[1] : '';
  var stack = <String>[];
  var ri = 0;
  for (var ch in bracket.split('')) {
    if (ch == '(' || ch == ',' || ch == ' ') {
      continue;
    } else if (ch == ')') {
      var right = stack.removeLast();
      var left = stack.removeLast();
      var result = ri < results.length ? results[ri++] : '0';
      stack.add(result == '0' ? left : right);
    } else {
      stack.add(ch);
    }
  }
  return stack.isNotEmpty ? stack.first : '';
}

@pragma('vm:entry-point')
void main() {
  assert(simulateTournament("((A,B),(C,D)):110") == "B");
  assert(simulateTournament("X:") == "X");
  assert(simulateTournament("(A,B):1") == "B");
  print('All tests passed!');
}