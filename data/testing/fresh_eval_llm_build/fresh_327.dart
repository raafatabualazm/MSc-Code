@pragma('vm:entry-point')
List<int> computeRoundedWindowSums(List<int> amounts) {
  if (amounts.length < 3) return [];
  List<int> result = [];
  for (int i = 0; i <= amounts.length - 3; i++) {
    int sum = amounts[i] + amounts[i+1] + amounts[i+2];
    result.add((sum + 2) ~/ 5 * 5);
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(computeRoundedWindowSums([2,3,4,5]).toString() == '[10, 10]');
  assert(computeRoundedWindowSums([1]).toString() == '[]');
  assert(computeRoundedWindowSums([100, 200, 300]).toString() == '[600]');
  print('All tests passed!');
}