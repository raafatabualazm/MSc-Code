@pragma('vm:entry-point')
String replayChessSquareStack(String script) {
  var stack = <String>[];
  if (script.trim().isEmpty) return 'empty';
  for (var token in script.split(' ')) {
    if (token.isEmpty) continue;
    var valid = token.length == 2;
    if (valid) {
      for (var i = 0; i < 2; i++) {
        var c = token.codeUnitAt(i);
        if ((i == 0 && (c < 97 || c > 104)) ||
            (i == 1 && (c < 49 || c > 56))) {
          valid = false;
          break;
        }
      }
    }
    if (valid) {
      stack.add(token);
    } else if (token == '^') {
      if (stack.isEmpty) return 'underflow';
      stack.removeLast();
    } else if (token == '~') {
      if (stack.isEmpty) continue;
      var top = stack.last;
      var f = 201 - top.codeUnitAt(0);
      var r = 105 - top.codeUnitAt(1);
      stack.add('${String.fromCharCode(f)}${String.fromCharCode(r)}');
    } else if (token == '!') {
      while (stack.length >= 2 && stack.last == stack[stack.length - 2]) {
        stack.removeLast();
      }
    } else {
      return 'invalid';
    }
  }
  return stack.isEmpty ? 'empty' : stack.join('>');
}

@pragma('vm:entry-point')
void main() {
  assert(replayChessSquareStack('a1 ~') == 'a1>h8');
  assert(replayChessSquareStack('a1 a1 !') == 'a1');
  assert(replayChessSquareStack('^') == 'underflow');
  print('All tests passed!');
}