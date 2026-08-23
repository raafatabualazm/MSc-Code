@pragma('vm:entry-point')
bool hasBalancedTelemetrySplits(List<int> samples) {
  bool check(List<int> part) {
    if (part.length <= 1) return true;
    int mid = part.length ~/ 2;
    int left = part.sublist(0, mid).fold(0, (a, b) => a + b);
    int right = part.sublist(mid).fold(0, (a, b) => a + b);
    return (left - right).abs() <= 1 &&
        check(part.sublist(0, mid)) &&
        check(part.sublist(mid));
  }
  return check(samples);
}

@pragma('vm:entry-point')
void main() {
  assert(hasBalancedTelemetrySplits([]) == true);
  assert(hasBalancedTelemetrySplits([1, 3]) == false);
  assert(hasBalancedTelemetrySplits([2, 1, 1]) == true);
  print('All tests passed!');
}