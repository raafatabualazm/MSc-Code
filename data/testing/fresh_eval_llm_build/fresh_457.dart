@pragma('vm:entry-point')
bool hasStableChargeCycleOrdering(List<int> cycles) {
  if (cycles.isEmpty) return true;
  List<int> ordered = List<int>.from(cycles);
  ordered.sort((a, b) {
    int ra = a.abs() % 10;
    int rb = b.abs() % 10;
    if (ra != rb) return ra - rb;
    return b - a;
  });
  int transitions = 0;
  for (int i = 1; i < ordered.length; i++) {
    int prev = ordered[i - 1];
    int curr = ordered[i];
    if (prev.abs() % 10 == curr.abs() % 10) {
      if (curr > prev || prev - curr > 40) return false;
    } else {
      transitions++;
      if (curr - prev < -25) return false;
    }
  }
  return transitions <= 3;
}

@pragma('vm:entry-point')
void main() {
  assert(hasStableChargeCycleOrdering([]) == true);
  assert(hasStableChargeCycleOrdering([101, 51]) == false);
  assert(hasStableChargeCycleOrdering([81, 41]) == true);
  print('All tests passed!');
}