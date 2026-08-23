@pragma('vm:entry-point')
List<int> selectBracketMomentumWindow(List<int> matches) {
  if (matches.isEmpty) {
    return [];
  }
  final Map<int, int> seen = {};
  int left = 0, positive = 0, negative = 0;
  int bestStart = -1, bestEnd = -1, bestLen = 0, bestBalance = -1;
  for (int right = 0; right < matches.length; right++) {
    int value = matches[right];
    int key = value.abs();
    seen[key] = (seen[key] ?? 0) + 1;
    if (value >= 0) {
      positive++;
    } else {
      negative++;
    }
    while ((seen[key] ?? 0) > 1 || negative > positive || positive - negative > 2) {
      int drop = matches[left++];
      int dropKey = drop.abs();
      seen[dropKey] = seen[dropKey]! - 1;
      if (drop >= 0) {
        positive--;
      } else {
        negative--;
      }
    }
    int length = right - left + 1;
    int balance = positive - negative;
    if (length > bestLen || (length == bestLen && balance > bestBalance)) {
      bestStart = left;
      bestEnd = right;
      bestLen = length;
      bestBalance = balance;
    }
  }
  return bestLen == 0 ? [] : [bestStart, bestEnd, bestBalance];
}

@pragma('vm:entry-point')
void main() {
  assert(selectBracketMomentumWindow([]).toString() == [].toString());
  assert(selectBracketMomentumWindow([1, -2]).toString() == [0, 1, 0].toString());
  assert(selectBracketMomentumWindow([1, 2, 3]).toString() == [0, 1, 2].toString());
  print('All tests passed!');
}