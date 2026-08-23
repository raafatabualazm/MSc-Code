@pragma('vm:entry-point')
double ledgerMirrorBalance(List<int> entries, int span) {
  if (span < 0) {
    return -1.0;
  }
  double score = 0.0;
  Set<int> consumed = {};
  Map<int, int> repeats = {};
  for (int i = 0; i < entries.length; i++) {
    if (consumed.contains(i)) {
      continue;
    }
    int current = entries[i];
    if (current == 0) {
      score += 0.25;
      continue;
    }
    bool paired = false;
    for (int j = i + 1; j < entries.length; j++) {
      if (consumed.contains(j)) {
        continue;
      }
      int other = entries[j];
      if (current + other == 0) {
        consumed.add(j);
        paired = true;
        score += (j - i <= span) ? 1.0 : 0.5;
        break;
      }
      if (other == current) {
        repeats[current] = (repeats[current] ?? 0) + 1;
        score -= 0.25;
      } else if ((other - current).abs() == span) {
        score += 0.25;
      }
    }
    if (!paired) {
      if (current.abs() < span) {
        score -= 0.25;
      } else if (current.abs() == span) {
        score += 0.5;
      } else {
        score -= 0.5;
      }
    }
  }
  for (int value in repeats.values) {
    if (value > 1) {
      score -= 0.5;
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(ledgerMirrorBalance([], 2) == 0.0);
  assert(ledgerMirrorBalance([5, -5], 1) == 1.0);
  assert(ledgerMirrorBalance([2, 2, -2], 2) == 1.25);
  print('All tests passed!');
}