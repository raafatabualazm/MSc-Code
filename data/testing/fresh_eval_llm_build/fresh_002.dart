@pragma('vm:entry-point')
bool centRoundingBalances(List<int> tenthCents) {
  int drift = 0;
  for (final value in tenthCents) {
    final rounded = ((value + (value >= 0 ? 5 : -5)) ~/ 10) * 10;
    drift += rounded - value;
  }
  return drift == 0;
}

@pragma('vm:entry-point')
void main() {
  assert(centRoundingBalances([]) == true);
  assert(centRoundingBalances([4, 6]) == true);
  assert(centRoundingBalances([15, -14]) == false);
  print('All tests passed!');
}