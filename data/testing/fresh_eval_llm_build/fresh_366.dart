@pragma('vm:entry-point')
int telemetryRollbackScore(List<String> events) {
  final List<int> samples = [];
  final List<List<int>> history = [];
  for (final e in events) {
    if (e.startsWith('S')) {
      history.add(List<int>.from(samples));
      samples.add(int.parse(e.substring(1)));
    } else if (e == 'F') {
      if (samples.isNotEmpty) {
        history.add(List<int>.from(samples));
        samples.removeLast();
      }
    } else if (e == 'M') {
      if (samples.length >= 2) {
        history.add(List<int>.from(samples));
        samples[samples.length - 2] += samples.removeLast();
      }
    } else if (e == 'U' && history.isNotEmpty) {
      samples
        ..clear()
        ..addAll(history.removeLast());
    }
  }
  int total = 0;
  for (final v in samples) {
    total += v.abs().isEven ? v ~/ 2 : v;
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(telemetryRollbackScore([]) == 0);
  assert(telemetryRollbackScore(['S2', 'S5', 'M']) == 7);
  assert(telemetryRollbackScore(['S8', 'F', 'U']) == 4);
  print('All tests passed!');
}