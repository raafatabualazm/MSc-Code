@pragma('vm:entry-point')
int telemetryTransitionBalance(List<int> samples) {
  List<int> filtered = [];
  for (int value in samples) {
    if (filtered.isEmpty || filtered.last != value) {
      filtered.add(value);
    }
  }
  int score = 0;
  for (int i = 0; i < filtered.length; i++) {
    int value = filtered[i];
    if (value < 0) {
      score -= 2;
      if (i > 0 && filtered[i - 1] >= 0) score += 1;
    } else if (value >= 50) {
      score += 3;
      if (i > 0 && filtered[i - 1] >= 0 && filtered[i - 1] < 50) score += 2;
    } else {
      score += 1;
      if (i > 0 && filtered[i - 1] >= 50) score -= 1;
    }
    if (i > 0) {
      int diff = (value - filtered[i - 1]).abs();
      if (diff == 1) score += 1;
      else if (diff > 20) score += 2;
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(telemetryTransitionBalance([]) == 0);
  assert(telemetryTransitionBalance([49, 50]) == 7);
  assert(telemetryTransitionBalance([51, 51, 30, -1, -1, 0]) == 8);
  print('All tests passed!');
}