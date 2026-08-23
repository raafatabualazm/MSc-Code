@pragma('vm:entry-point')
bool validateBarcodeUndoFrames(String tape) {
  List<int> stack = [];
  for (int i = 0; i < tape.length; i++) {
    String ch = tape[i];
    if (ch.codeUnitAt(0) >= 48 && ch.codeUnitAt(0) <= 57) {
      stack.add(ch.codeUnitAt(0) - 48);
    } else if (ch == '<') {
      if (stack.isEmpty || stack.last == -1) return false;
      stack.removeLast();
    } else if (ch == '[') {
      stack.add(-1);
    } else if (ch == ']') {
      int sum = 0, count = 0;
      List<int> seg = [];
      while (stack.isNotEmpty && stack.last != -1) {
        int v = stack.removeLast();
        seg.add(v);
        sum += v;
        count++;
      }
      if (stack.isEmpty || count < 3 || sum % 10 != count) return false;
      stack.removeLast();
      for (int j = 1; j < seg.length; j++) {
        if ((seg[j] & 1) == (seg[j - 1] & 1)) return false;
      }
      stack.add(count);
    } else {
      return false;
    }
  }
  for (int v in stack) {
    if (v == -1) return false;
  }
  return stack.length == 1 && stack[0].isOdd;
}

@pragma('vm:entry-point')
void main() {
  assert(validateBarcodeUndoFrames("[012]") == true);
  assert(validateBarcodeUndoFrames("[123]") == false);
  assert(validateBarcodeUndoFrames("[012]4<") == true);
  print('All tests passed!');
}