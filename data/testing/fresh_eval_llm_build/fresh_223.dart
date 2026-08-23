@pragma('vm:entry-point')
String auditRgbDaySpans(List<int> pixels, int epochDay) {
  if (pixels.isEmpty) return 'empty';
  int earliest = 1 << 30;
  int latest = -(1 << 30);
  int score = 0;
  int flagged = 0;
  for (int i = 0; i < pixels.length; i++) {
    int p = pixels[i];
    List<int> channels = [(p >> 16) & 255, (p >> 8) & 255, p & 255];
    int base = epochDay + (i * 2);
    for (int j = 0; j < channels.length; j++) {
      int v = channels[j];
      if (v == 0) continue;
      int span = (v % 5) + 1;
      int start = base + j;
      int end = start + span - 1;
      if (start < earliest) earliest = start;
      if (end > latest) latest = end;
      score += v.isEven ? span : -span;
      for (int d = start; d <= end; d++) {
        if (d < 0) continue;
        if (((d + v + j) % 4) == 0) {
          flagged++;
        } else if (((d ^ v) & 1) == 0) {
          score++;
        } else {
          score--;
        }
      }
    }
  }
  if (latest < earliest) return 'void';
  return '$earliest:$latest:${latest - earliest + 1}:$score:$flagged';
}

@pragma('vm:entry-point')
void main() {
  assert(auditRgbDaySpans([], 0) == 'empty');
  assert(auditRgbDaySpans([0], 5) == 'void');
  assert(auditRgbDaySpans([1], 0) == '2:3:2:-2:0');
  print('All tests passed!');
}