@pragma('vm:entry-point')
int countStableTelemetryBursts(String stream, int limit) {
  int count = 0;
  int value = 0;
  bool reading = false, invalid = false, muted = false, hasDigit = false;
  for (int i = 0; i < stream.length; i++) {
    String c = stream[i];
    if (!reading) {
      if (c == 'T') {
        reading = true; value = 0; invalid = false; muted = false; hasDigit = false;
      }
    } else if (c.codeUnitAt(0) >= 48 && c.codeUnitAt(0) <= 57) {
      value = value * 10 + c.codeUnitAt(0) - 48; hasDigit = true;
    } else if (c == '#' && hasDigit && !muted) {
      muted = true;
    } else if (c == ';') {
      if (!invalid && hasDigit && !muted && value <= limit) count++;
      reading = false;
    } else {
      invalid = true;
    }
  }
  return count;
}

@pragma('vm:entry-point')
void main() {
  assert(countStableTelemetryBursts('', 3) == 0);
  assert(countStableTelemetryBursts('T4;T5#;T2;', 4) == 2);
  assert(countStableTelemetryBursts('T9;T10;T11;', 10) == 2);
  print('All tests passed!');
}