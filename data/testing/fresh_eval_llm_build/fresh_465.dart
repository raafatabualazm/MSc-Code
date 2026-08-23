@pragma('vm:entry-point')
List<int> longestBalancedChessWindow(List<String> squares) {
  int left = 0, dark = 0, light = 0, bestStart = 0, bestLen = 0;
  final seen = <String, int>{};
  for (int right = 0; right < squares.length; right++) {
    String s = squares[right];
    seen[s] = (seen[s] ?? 0) + 1;
    int file = s.codeUnitAt(0) - 96, rank = s.codeUnitAt(1) - 48;
    if ((file + rank).isEven) {
      dark++;
    } else {
      light++;
    }
    while (seen[s]! > 1 || (dark - light).abs() > 1) {
      String t = squares[left++];
      seen[t] = seen[t]! - 1;
      int tf = t.codeUnitAt(0) - 96, tr = t.codeUnitAt(1) - 48;
      if ((tf + tr).isEven) {
        dark--;
      } else {
        light--;
      }
    }
    if (right - left + 1 > bestLen) {
      bestLen = right - left + 1;
      bestStart = left;
    }
  }
  return [bestStart, bestLen];
}

@pragma('vm:entry-point')
void main() {
  assert(longestBalancedChessWindow(['a1']).toString() == '[0, 1]');
  assert(longestBalancedChessWindow(['a1', 'b1', 'c1']).toString() == '[0, 3]');
  assert(longestBalancedChessWindow(['a1', 'c1', 'e1', 'b1']).toString() == '[2, 2]');
  print('All tests passed!');
}