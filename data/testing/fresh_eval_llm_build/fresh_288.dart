@pragma('vm:entry-point')
int sumTidalSurges(List<int> heights) {
  int total = 0;
  for (int i = 3; i < heights.length - 3; i++) {
    int sum = 0;
    for (int j = i - 3; j <= i + 3; j++) {
      sum += heights[j];
    }
    int avg = sum ~/ 7;
    if (heights[i] > 0 && heights[i] * 4 > avg * 5) {
      total += heights[i];
    }
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(sumTidalSurges([]) == 0);
  assert(sumTidalSurges([5,5,5,20,5,5,5]) == 20);
  assert(sumTidalSurges([8,8,8,10,8,8,8]) == 0);
  print('All tests passed!');
}