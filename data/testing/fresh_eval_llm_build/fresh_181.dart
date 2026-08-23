@pragma('vm:entry-point')
List<num> mostFrequentFloors(List<int> requests) {
  var freq = <int, int>{};
  var top = <int>[];
  int max = 0;
  for (var f in requests) {
    int c = (freq[f] ?? 0) + 1;
    freq[f] = c;
    if (c > max) {
      max = c;
      top = [f];
    } else if (c == max) {
      top.add(f);
    }
  }
  top.sort();
  return top;
}

@pragma('vm:entry-point')
void main() {
  assert(mostFrequentFloors([1,2,2,1,3]).toString() == '[1, 2]');
  assert(mostFrequentFloors([]).toString() == '[]');
  assert(mostFrequentFloors([5]).toString() == '[5]');
  print('All tests passed!');
}