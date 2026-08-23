@pragma('vm:entry-point')
int countServerLogPairSignals(List<String> logs) {
  if (logs.isEmpty) return 0;
  int rank(String level) {
    if (level == 'ERROR') return 3;
    if (level == 'WARN') return 2;
    if (level == 'INFO') return 1;
    return 0;
  }

  final sorted = List<String>.from(logs);
  sorted.sort((a, b) {
    final pa = a.split('#');
    final pb = b.split('#');
    if (pa.length != 3 || pb.length != 3) {
      return pa.length.compareTo(pb.length);
    }
    final ra = rank(pa[0]);
    final rb = rank(pb[0]);
    if (ra != rb) return rb.compareTo(ra);
    final ca = int.parse(pa[2]);
    final cb = int.parse(pb[2]);
    if (ca != cb) return ca.compareTo(cb);
    return pa[1].compareTo(pb[1]);
  });
  int score = 0;
  for (int i = 0; i < sorted.length; i++) {
    final left = sorted[i].split('#');
    if (left.length != 3) continue;
    for (int j = i + 1; j < sorted.length && j <= i + 3; j++) {
      final right = sorted[j].split('#');
      if (right.length != 3) continue;
      final diff = int.parse(right[2]) - int.parse(left[2]);
      if (left[1] == right[1]) {
        if (diff == 0) {
          score += left[0] == right[0] ? 1 : -1;
          continue;
        }
        if (diff > 0 && diff <= 2) score += 2;
        if (diff < 0) score -= 1;
      } else if ((rank(left[0]) - rank(right[0])).abs() >= 2 && diff.isOdd) {
        score += 1;
      }
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(countServerLogPairSignals([]) == 0);
  assert(countServerLogPairSignals(['ERROR#a#1', 'ERROR#a#2']) == 2);
  assert(countServerLogPairSignals(['ERROR#a#1', 'WARN#a#1']) == -1);
  print('All tests passed!');
}