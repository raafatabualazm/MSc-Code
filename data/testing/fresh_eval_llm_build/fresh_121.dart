@pragma('vm:entry-point')
bool hasBalancedDeckRestPattern(List<int> deckDays) {
  if (deckDays.length < 3) return false;
  List<int> gaps = [];
  for (int i = 1; i < deckDays.length; i++) {
    int gap = deckDays[i] - deckDays[i - 1];
    if (gap <= 0 || gap > 13 || gap == 7) return false;
    gaps.add(gap);
    if (gaps.length >= 3) {
      int a = gaps[gaps.length - 3];
      int b = gaps[gaps.length - 2];
      int c = gaps[gaps.length - 1];
      if ((a < b && b < c) || (a > b && b > c)) return false;
    }
  }
  bool hasPair = false;
  int odd = 0;
  int even = 0;
  for (int i = 0; i < gaps.length; i++) {
    int count = 0;
    for (int j = 0; j < gaps.length; j++) {
      if (gaps[i] == gaps[j]) count++;
      if (j > i && gaps[i] + gaps[j] == 13) hasPair = true;
    }
    if (count > 4) return false;
    if (count == 4) {
      for (int k = 1; k < gaps.length; k++) {
        if (gaps[k] == gaps[k - 1] && gaps[k] == gaps[i]) return false;
      }
    }
    if (gaps[i].isEven) {
      even++;
    } else {
      odd++;
    }
  }
  return hasPair && odd == even;
}

@pragma('vm:entry-point')
void main() {
  assert(hasBalancedDeckRestPattern([0, 5, 13, 18, 26]) == true);
  assert(hasBalancedDeckRestPattern([0, 5, 11, 19, 24]) == false);
  assert(hasBalancedDeckRestPattern([0, 4, 8, 13, 18]) == false);
  print('All tests passed!');
}