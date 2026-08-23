@pragma('vm:entry-point')
List<int> findPassWindowGaps(List<int> passes) {
  if (passes.isEmpty) return [];
  int n = passes.length;
  List<List<int>> intervals = [];
  for (int i = 0; i < n; i += 2) {
    intervals.add([passes[i], passes[i+1]]);
  }
  intervals.sort((a, b) => a[0].compareTo(b[0]));
  List<List<int>> merged = [];
  for (var iv in intervals) {
    if (merged.isEmpty || merged.last[1] < iv[0] - 1) {
      merged.add([iv[0], iv[1]]);
    } else {
      if (iv[1] > merged.last[1]) merged.last[1] = iv[1];
    }
  }
  List<int> gaps = [];
  if (merged.isNotEmpty && merged[0][0] > 0) {
    gaps.add(0);
    gaps.add(merged[0][0] - 1);
  }
  for (int i = 1; i < merged.length; i++) {
    int gapStart = merged[i-1][1] + 1;
    int gapEnd = merged[i][0] - 1;
    if (gapStart <= gapEnd) {
      gaps.add(gapStart);
      gaps.add(gapEnd);
    }
  }
  return gaps;
}

@pragma('vm:entry-point')
void main() {
  assert(findPassWindowGaps([2,5,8,10]).toString() == '[0, 1, 6, 7]');
  assert(findPassWindowGaps([0,3,7,9]).toString() == '[4, 6]');
  assert(findPassWindowGaps([]).toString() == '[]');
  print('All tests passed!');
}