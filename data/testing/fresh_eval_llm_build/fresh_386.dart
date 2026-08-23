@pragma('vm:entry-point')
String theaterRowCycleCode(List<int> seatCounts) {
  int gcd(int a, int b) => b == 0 ? a : gcd(b, a % b);
  int g = 0;
  for (final seats in seatCounts) {
    if (seats != 0) {
      g = g == 0 ? seats.abs() : gcd(g, seats.abs());
    }
  }
  return g == 0 ? 'empty' : g.toRadixString(8);
}

@pragma('vm:entry-point')
void main() {
  assert(theaterRowCycleCode([12, 18]) == '6');
  assert(theaterRowCycleCode([64, 32, 16]) == '20');
  assert(theaterRowCycleCode([0, 0]) == 'empty');
  print('All tests passed!');
}