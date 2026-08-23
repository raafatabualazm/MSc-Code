@pragma('vm:entry-point')
int telemetryCascadeChecksum(List<int> samples) {
  int scan(int start, int end) {
    if (start > end) return 0;
    if (start == end) {
      int v = samples[start];
      return v == 0 ? 1 : (v > 0 ? v : -v) % 5;
    }
    int mid = (start + end) >> 1;
    int score = scan(start, mid) + scan(mid + 1, end);
    for (int i = start; i <= end; i++) {
      int v = samples[i];
      if (i > start && v == samples[i - 1]) {
        score -= 1;
      } else if (v > 7 || v < -7) {
        score += 3;
      } else if (v == 0 && (i == end || samples[i + 1] != 0)) {
        score += 2;
      }
    }
    return score + (samples[mid] >= 0 ? 1 : -1);
  }
  return scan(0, samples.length - 1);
}

@pragma('vm:entry-point')
void main() {
  assert(telemetryCascadeChecksum([]) == 0);
  assert(telemetryCascadeChecksum([1, 2]) == 4);
  assert(telemetryCascadeChecksum([8, -8]) == 13);
  print('All tests passed!');
}