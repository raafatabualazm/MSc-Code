@pragma('vm:entry-point')
List<int> sortPassWindowsByAverageDuration(List<int> durations) {
  if (durations.isEmpty) return [];
  int total = durations.fold(0, (a, b) => a + b);
  double avg = total / durations.length;
  List<int> result = List<int>.from(durations);
  result.sort((a, b) {
    bool aBelow = a < avg, bBelow = b < avg;
    if (aBelow && bBelow) return b.compareTo(a);
    if (!aBelow && !bBelow) return a.compareTo(b);
    return aBelow ? -1 : 1;
  });
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(sortPassWindowsByAverageDuration([]).toString() == '[]');
  assert(sortPassWindowsByAverageDuration([5,1,9,3]).toString() == '[3, 1, 5, 9]');
  assert(sortPassWindowsByAverageDuration([6,4,2]).toString() == '[2, 4, 6]');
  print('All tests passed!');
}