@pragma('vm:entry-point')
int telemetryOrderingPenalty(List<int> samples) {
  List<int> ordered = [];
  for (int v in samples) {
    if (v == 0) continue;
    int pos = ordered.length;
    while (pos > 0) {
      int leftKey = ordered[pos - 1].abs() % 7;
      int rightKey = v.abs() % 7;
      if (leftKey > rightKey ||
          (leftKey == rightKey && ordered[pos - 1] < v)) {
        pos--;
      } else {
        break;
      }
    }
    ordered.insert(pos, v);
  }
  if (ordered.isEmpty) return 0;
  int score = 0;
  for (int i = 0; i < ordered.length; i++) {
    int repeats = 0;
    for (int j = 0; j < i; j++) {
      if (ordered[j].abs() == ordered[i].abs()) repeats++;
    }
    if (repeats >= 2) continue;
    if (i == 0) {
      score += ordered[i].isEven ? 1 : -1;
    } else if (ordered[i - 1].abs() % 7 == ordered[i].abs() % 7 &&
        ordered[i - 1].sign != ordered[i].sign) {
      score += 4;
    } else if (ordered[i].abs() > ordered[i - 1].abs()) {
      score += 2 + repeats;
    } else {
      score -= 1;
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(telemetryOrderingPenalty([]) == 0);
  assert(telemetryOrderingPenalty([1, -8]) == 3);
  assert(telemetryOrderingPenalty([2, -2, 2]) == 0);
  print('All tests passed!');
}