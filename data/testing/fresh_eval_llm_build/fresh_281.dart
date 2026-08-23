@pragma('vm:entry-point')
int countStrongEvenRounds(List<List<int>> rounds) {
  int count = 0;
  for (var round in rounds) {
    Map<int, int> freq = {};
    int sum = 0;
    for (var die in round) {
      freq[die] = (freq[die] ?? 0) + 1;
      sum += die;
    }
    int maxFreq = freq.values.fold(0, (a, b) => a > b ? a : b);
    if (maxFreq >= 3 && sum % 2 == 0) {
      count++;
    }
  }
  return count;
}

@pragma('vm:entry-point')
void main() {
  assert(countStrongEvenRounds([]) == 0);
  assert(countStrongEvenRounds([[2,2,2]]) == 1);
  assert(countStrongEvenRounds([[1,1,1], [2,2,2]]) == 1);
  print('All tests passed!');
}