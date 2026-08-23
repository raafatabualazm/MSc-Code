@pragma('vm:entry-point')
String encodeTelemetryDrift(List<int> samples) {
  if (samples.isEmpty) return 'idle';
  String out = '';
  for (int i = 0; i < samples.length; i++) {
    int n = samples[i].abs();
    if (n == 0) { out += 'z'; continue; }
    int a = n, b = i == 0 ? 1 : samples[i - 1].abs();
    while (b != 0) {
      int t = a % b;
      a = b;
      b = t;
    }
    int factors = 0, temp = n;
    for (int d = 2; d * d <= temp; d += d == 2 ? 1 : 2) {
      bool used = false;
      while (temp % d == 0) {
        temp ~/= d;
        used = true;
      }
      if (used) factors++;
    }
    if (temp > 1) factors++;
    int code = (a + factors + n) % 16;
    if (n % 2 == 0 && factors > 1) {
      out += code.toRadixString(16);
    } else if (a == 1) {
      out += String.fromCharCode(97 + code);
    } else {
      out += code.toString();
    }
  }
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(encodeTelemetryDrift([]) == 'idle');
  assert(encodeTelemetryDrift([6]) == '9');
  assert(encodeTelemetryDrift([-10, 15]) == 'd6');
  print('All tests passed!');
}