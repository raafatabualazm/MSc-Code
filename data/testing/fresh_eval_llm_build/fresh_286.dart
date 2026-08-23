@pragma('vm:entry-point')
double packetBurstScore(List<String> log) {
  List<int> stack = [];
  for (String token in log) {
    if (token.startsWith('P')) {
      stack.add(int.parse(token.substring(1)));
    } else if (token == 'MERGE') {
      if (stack.length >= 2) {
        int merged = stack.removeLast() + stack.removeLast();
        stack.add(merged > 16 ? 16 : merged);
      }
    } else if (token == 'RETRY') {
      if (stack.isNotEmpty) {
        int top = stack.removeLast();
        if (top.isOdd) stack.add(top + 1);
        stack.add(top ~/ 2);
      }
    } else if (stack.isNotEmpty) {
      if (stack.last <= 4) {
        stack.removeLast();
      } else {
        stack[stack.length - 1] -= 4;
      }
    }
  }
  double score = 0.0;
  for (int size in stack) {
    score += size >= 12 ? size / 2.0 : size / 4.0;
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(packetBurstScore([]) == 0.0);
  assert(packetBurstScore(['P8', 'DROP']) == 1.0);
  assert(packetBurstScore(['P7', 'P5', 'MERGE']) == 6.0);
  print('All tests passed!');
}