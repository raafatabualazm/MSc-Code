@pragma('vm:entry-point')
bool hasBalancedTideDirectionChanges(List<int> readings) {
  int rises = 0;
  int falls = 0;
  for (int i = 1; i < readings.length; i++) {
    if (readings[i] == readings[i - 1]) continue;
    if (readings[i] > readings[i - 1]) {
      rises++;
    } else {
      falls++;
    }
  }
  return rises == falls && rises + falls > 0;
}

@pragma('vm:entry-point')
void main() {
  assert(hasBalancedTideDirectionChanges([1, 2, 1]) == true);
  assert(hasBalancedTideDirectionChanges([4, 4, 4]) == false);
  assert(hasBalancedTideDirectionChanges([2, 3, 4, 3]) == false);
  print('All tests passed!');
}