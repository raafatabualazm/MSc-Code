@pragma('vm:entry-point')
String inventorySigilCode(List<int> crates) {
  int common = 0;
  int rarity = 0;
  for (final raw in crates) {
    int v = raw.abs();
    int a = common, b = v;
    while (b != 0) {
      int t = a % b;
      a = b;
      b = t;
    }
    common = a;
    if (v > 1) {
      bool prime = true;
      for (int d = 2; d * d <= v; d++) {
        if (v % d == 0) {
          prime = false;
          break;
        }
      }
      rarity += prime ? v : -v;
    }
  }
  if (crates.isEmpty) return 'empty';
  int seal = common.abs() + rarity.abs();
  return '${common}:${seal.toRadixString(16)}';
}

@pragma('vm:entry-point')
void main() {
  assert(inventorySigilCode([]) == 'empty');
  assert(inventorySigilCode([2, 4]) == '2:4');
  assert(inventorySigilCode([8, 12, 16]) == '4:28');
  print('All tests passed!');
}