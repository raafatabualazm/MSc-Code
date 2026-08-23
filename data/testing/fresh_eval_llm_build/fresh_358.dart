@pragma('vm:entry-point')
int passwordCursorRadiusViolations(String moves, int limit) {
  int x = 0, y = 0, violations = 0;
  for (final c in moves.split('')) {
    if (c == 'U') y++;
    else if (c == 'D') y--;
    else if (c == 'L') x--;
    else if (c == 'R') x++;
    if (x.abs() + y.abs() > limit) violations++;
  }
  return violations;
}

@pragma('vm:entry-point')
void main() {
  assert(passwordCursorRadiusViolations("", 0) == 0);
  assert(passwordCursorRadiusViolations("RU", 1) == 1);
  assert(passwordCursorRadiusViolations("UUURRR", 3) == 3);
  print('All tests passed!');
}