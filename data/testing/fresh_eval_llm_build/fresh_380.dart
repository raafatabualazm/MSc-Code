@pragma('vm:entry-point')
String findHalfCycleDay(List<int> cumulativeCycles) {
  if (cumulativeCycles.isEmpty) return "0";
  int total = cumulativeCycles.last;
  int half = (total + 1) ~/ 2;
  int low = 0, high = cumulativeCycles.length - 1;
  while (low < high) {
    int mid = (low + high) ~/ 2;
    if (cumulativeCycles[mid] >= half) {
      high = mid;
    } else {
      low = mid + 1;
    }
  }
  return (low + 1).toString();
}

@pragma('vm:entry-point')
void main() {
  assert(findHalfCycleDay([]) == "0");
  assert(findHalfCycleDay([5]) == "1");
  assert(findHalfCycleDay([2, 3, 5]) == "2");
  print('All tests passed!');
}