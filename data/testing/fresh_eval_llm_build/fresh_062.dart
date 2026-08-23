@pragma('vm:entry-point')
int minDoublingsToMeetIngredientThresholds(List<int> quantities) {
  if (quantities.isEmpty) return 0;
  for (int q in quantities) {
    if (q <= 0) return -1;
  }
  int doublings = 0;
  while (doublings <= 60) {
    bool allMet = true;
    for (int i = 0; i < quantities.length; i++) {
      int scaled = quantities[i] << doublings;
      int threshold = (i + 1) * 3;
      if (scaled < threshold) {
        allMet = false;
        break;
      }
    }
    if (allMet) return doublings;
    doublings++;
  }
  return -1;
}

@pragma('vm:entry-point')
void main() {
  assert(minDoublingsToMeetIngredientThresholds([]) == 0);
  assert(minDoublingsToMeetIngredientThresholds([1, 1, 1]) == 4);
  assert(minDoublingsToMeetIngredientThresholds([3, 6, 9]) == 0);
  print('All tests passed!');
}