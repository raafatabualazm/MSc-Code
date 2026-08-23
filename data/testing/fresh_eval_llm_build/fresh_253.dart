@pragma('vm:entry-point')
bool followsTelemetryMagnitudeCadence(List<int> samples) {
  List<int> ordered = List<int>.from(samples);
  ordered.sort((a, b) {
    int aa = a.abs();
    int bb = b.abs();
    if (aa != bb) {
      return aa - bb;
    }
    return b - a;
  });
  int risingPairs = 0;
  for (int i = 1; i < ordered.length; i++) {
    int diff = ordered[i] - ordered[i - 1];
    if (diff == 0) {
      return false;
    } else if (diff > 0) {
      risingPairs++;
      if (risingPairs > 2 && ordered[i].isEven) {
        return false;
      }
    } else if (ordered[i].abs() == ordered[i - 1].abs()) {
      return false;
    }
  }
  return ordered.length < 2 || risingPairs >= ordered.length ~/ 2;
}

@pragma('vm:entry-point')
void main() {
  assert(followsTelemetryMagnitudeCadence([]) == true);
  assert(followsTelemetryMagnitudeCadence([1, -1]) == false);
  assert(followsTelemetryMagnitudeCadence([1, 2, 3, 5]) == true);
  print('All tests passed!');
}