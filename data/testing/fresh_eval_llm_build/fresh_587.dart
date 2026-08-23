@pragma('vm:entry-point')
num telemetryRollbackDrift(String log) {
  final samples = <int>[];
  final history = <List<int>>[];
  for (final raw in log.split(',')) {
    final token = raw.trim();
    if (token.isEmpty) continue;
    if (token == 'UNDO') {
      if (history.isNotEmpty) {
        samples
          ..clear()
          ..addAll(history.removeLast());
      }
      continue;
    }
    history.add(List<int>.from(samples));
    final value = int.tryParse(token);
    if (value != null) {
      samples.add(value);
      continue;
    }
    if (token == 'PAIR') {
      if (samples.length < 2) return -1;
      final a = samples.removeLast();
      final b = samples.removeLast();
      samples..add(b - a)..add(a + b);
    } else if (token.startsWith('DROP')) {
      final limit = int.parse(token.substring(4));
      var removed = 0;
      while (samples.isNotEmpty && removed < limit) {
        if (samples.last <= 0) break;
        samples.removeLast();
        removed++;
      }
    } else if (token == 'MERGE') {
      for (var i = samples.length - 1; i > 0; i--) {
        if (samples[i] == samples[i - 1]) {
          samples[i - 1] *= 2;
          samples.removeAt(i);
        }
      }
    } else {
      history.removeLast();
    }
  }
  num total = 0;
  for (final v in samples) {
    total += v;
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(telemetryRollbackDrift('5,3,PAIR') == 10);
  assert(telemetryRollbackDrift('2,2,2,MERGE') == 6);
  assert(telemetryRollbackDrift('PAIR') == -1);
  print('All tests passed!');
}