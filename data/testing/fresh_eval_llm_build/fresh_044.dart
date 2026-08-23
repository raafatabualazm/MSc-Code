@pragma('vm:entry-point')
String summarizeTelemetryMasks(List<int> samples) {
  if (samples.isEmpty) return 'none';
  var out = StringBuffer();
  for (var i = 0; i < samples.length; i++) {
    var raw = samples[i];
    if (out.isNotEmpty) out.write('|');
    if (raw < 0) {
      out.write('N');
      continue;
    }
    var value = raw & 255, pop = 0, longest = 0, current = 0;
    for (var bit = 0; bit < 8; bit++) {
      if (((value >> bit) & 1) == 1) {
        pop++;
        current++;
        if (current > longest) longest = current;
      } else {
        current = 0;
      }
    }
    var rotated = ((value << 1) & 255) | (value >> 7);
    if (pop == 0) {
      out.write('quiet');
    } else if (pop >= 6 && (rotated & 3) == 3) {
      out.write('alarm');
    } else if (longest >= 3) {
      out.write('streak');
    } else if ((pop & 1) == 0) {
      out.write('even');
    } else {
      out.write('odd');
    }
    if (value == 255 || value == 127) out.write('!');
  }
  return out.toString();
}

@pragma('vm:entry-point')
void main() {
  assert(summarizeTelemetryMasks([]) == 'none');
  assert(summarizeTelemetryMasks([223]) == 'alarm');
  assert(summarizeTelemetryMasks([255, 127]) == 'alarm!|streak!');
  print('All tests passed!');
}