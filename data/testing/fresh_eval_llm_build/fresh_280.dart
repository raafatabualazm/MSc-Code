@pragma('vm:entry-point')
String? processTideLog(List<String> entries) {
  List<int> stack = [];
  for (final entry in entries) {
    if (entry == 'U') {
      if (stack.isEmpty) return null;
      stack.removeLast();
    } else if (entry == 'D') {
      if (stack.isEmpty) return null;
      stack.add(stack.last);
    } else if (entry == 'S') {
      if (stack.length < 2) return null;
      final top = stack.removeLast(), second = stack.removeLast();
      stack.add(top); stack.add(second);
    } else if (entry == 'R') {
      final temp = <int>[];
      while (stack.isNotEmpty) temp.add(stack.removeLast());
      stack = temp;
    } else {
      final value = int.tryParse(entry);
      if (value == null) return null;
      stack.add(value);
    }
  }
  return stack.isEmpty ? null : stack.join(',');
}

@pragma('vm:entry-point')
void main() {
  assert(processTideLog(["5"]) == "5");
  assert(processTideLog(["U"]) == null);
  assert(processTideLog(["3", "D", "U"]) == "3");
  print('All tests passed!');
}