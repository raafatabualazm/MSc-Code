@pragma('vm:entry-point')
List<int> wifiSignalWindowBins(List<int> bins, int allowedGap) {
  List<int> result = [];
  List<int> maxQ = [];
  List<int> minQ = [];
  int left = 0;
  for (int right = 0; right < bins.length; right++) {
    while (maxQ.isNotEmpty && bins[maxQ.last] < bins[right]) {
      maxQ.removeLast();
    }
    while (minQ.isNotEmpty && bins[minQ.last] > bins[right]) {
      minQ.removeLast();
    }
    maxQ.add(right);
    minQ.add(right);
    while (bins[maxQ.first] - bins[minQ.first] > allowedGap) {
      if (maxQ.first == left) maxQ.removeAt(0);
      if (minQ.first == left) minQ.removeAt(0);
      left++;
    }
    result.add(right - left + 1);
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(wifiSignalWindowBins([2, 5, 4], 3).toString() == '[1, 2, 3]');
  assert(wifiSignalWindowBins([1, 4], 2).toString() == '[1, 1]');
  assert(wifiSignalWindowBins([], 1).length == 0);
  print('All tests passed!');
}