@pragma('vm:entry-point')
int widestClearViewSection(List<int> seats, int maxBlocked) {
  if (seats.isEmpty || maxBlocked < 0) return 0;
  final Map<int, int> freq = {};
  int left = 0, blocked = 0, repeated = 0, badPairs = 0, best = 0;
  for (int right = 0; right < seats.length; right++) {
    int v = seats[right];
    if (v < 0) {
      blocked++;
    } else {
      freq[v] = (freq[v] ?? 0) + 1;
      if (freq[v] == 3) repeated++;
    }
    if (right > 0 && seats[right - 1] >= 0 && v >= 0 && (seats[right - 1] - v).abs() > 3) {
      badPairs++;
    }
    while (blocked > maxBlocked || repeated > 0 || badPairs > 0) {
      if (left < right && seats[left] >= 0 && seats[left + 1] >= 0 && (seats[left] - seats[left + 1]).abs() > 3) {
        badPairs--;
      }
      int drop = seats[left++];
      if (drop < 0) {
        blocked--;
      } else {
        if (freq[drop] == 3) repeated--;
        freq[drop] = freq[drop]! - 1;
        if (freq[drop] == 0) freq.remove(drop);
      }
    }
    int span = right - left + 1;
    if (span > best) best = span;
  }
  return best;
}

@pragma('vm:entry-point')
void main() {
  assert(widestClearViewSection([], 2) == 0);
  assert(widestClearViewSection([1, -1, 2, 3], 1) == 4);
  assert(widestClearViewSection([5, 5, 5, 2], 0) == 3);
  print('All tests passed!');
}