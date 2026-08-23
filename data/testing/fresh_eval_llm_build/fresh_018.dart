@pragma('vm:entry-point')
bool isPasswordGridValid(List<List<String>> grid) {
  if (grid.isEmpty) return false;
  bool hasUpper = false;
  bool isSpecial(String c) => c.length == 1 && !RegExp(r'[a-zA-Z0-9]').hasMatch(c);
  for (int r = 0; r < grid.length; r++) {
    bool rowHasDigit = false;
    for (int c = 0; c < grid[r].length; c++) {
      String ch = grid[r][c];
      if (ch.length == 1 && ch.codeUnitAt(0) >= 48 && ch.codeUnitAt(0) <= 57) rowHasDigit = true;
      if (ch.length == 1 && ch.codeUnitAt(0) >= 65 && ch.codeUnitAt(0) <= 90) hasUpper = true;
      if (isSpecial(ch)) {
        if (c + 1 < grid[r].length && isSpecial(grid[r][c + 1])) return false;
        if (r + 1 < grid.length && c < grid[r + 1].length && isSpecial(grid[r + 1][c])) return false;
      }
    }
    if (!rowHasDigit) return false;
  }
  return hasUpper;
}

@pragma('vm:entry-point')
void main() {
  assert(isPasswordGridValid([]) == false);
  assert(isPasswordGridValid([['A','1'],['2','!']]) == true);
  assert(isPasswordGridValid([['a','1'],['2','!']]) == false);
  print('All tests passed!');
}