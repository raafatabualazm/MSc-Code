@pragma('vm:entry-point')
int tallyRoundedTillCents(List<List<int>> trays) {
  int total = 0;
  for (final row in trays) {
    int rowRounded = 0;
    for (final units in row) {
      int base = units ~/ 4;
      int rem = units.abs() % 4;
      if (units >= 0) {
        rowRounded += rem >= 2 ? base + 1 : base;
      } else {
        rowRounded += rem >= 2 ? base - 1 : base;
      }
    }
    if (rowRounded > 0 && rowRounded.isOdd) {
      total += rowRounded - 1;
    } else if (rowRounded < 0 && rowRounded.isOdd) {
      total += rowRounded + 1;
    } else {
      total += rowRounded;
    }
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(tallyRoundedTillCents([]) == 0);
  assert(tallyRoundedTillCents([[2, 2]]) == 2);
  assert(tallyRoundedTillCents([[-6, -2]]) == -2);
  print('All tests passed!');
}