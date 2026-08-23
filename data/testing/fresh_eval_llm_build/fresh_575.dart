@pragma('vm:entry-point')
int widestRecountParitySpan(List<int> precincts) {
  if (precincts.isEmpty) return 0;
  int left = 0;
  int best = 0;
  int north = 0;
  int south = 0;
  for (int right = 0; right < precincts.length; right++) {
    int change = precincts[right];
    if (change > 0) {
      north += change;
    } else if (change < 0) {
      south += -change;
    } else {
      continue;
    }
    while (left <= right && (north - south).abs() > 2) {
      int drop = precincts[left];
      if (drop > 0) {
        north -= drop;
      } else if (drop < 0) {
        south += drop;
      }
      left++;
      while (left <= right && precincts[left] == 0) {
        left++;
      }
    }
    int span = right - left + 1;
    if (north >= 5 && south >= 5 && span > best) {
      best = span;
    }
  }
  return best;
}

@pragma('vm:entry-point')
void main() {
  assert(widestRecountParitySpan([]) == 0);
  assert(widestRecountParitySpan([2, -2, 2, -2, 1, -1]) == 6);
  assert(widestRecountParitySpan([2, -2, 2, -2, 1]) == 0);
  print('All tests passed!');
}