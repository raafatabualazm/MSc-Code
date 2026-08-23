@pragma('vm:entry-point')
List<int> calculateAdjacentBinCounts(List<int> quantities) {
  int n = quantities.length;
  List<int> result = List.filled(n, 0);
  for (int i = 0; i < n; i++) {
    int count = 0;
    for (int j = 0; j < n; j++) {
      if (i == j) continue;
      int a = quantities[i];
      int b = quantities[j];
      while (b != 0) {
        int t = a % b;
        a = b;
        b = t;
      }
      if (a == 1) {
        count++;
      }
    }
    result[i] = count;
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(calculateAdjacentBinCounts([2,3,5]).toString() == '[2, 2, 2]');
  assert(calculateAdjacentBinCounts([4,6,9]).toString() == '[0, 0, 0]');
  print('All tests passed!');
}