@pragma('vm:entry-point')
List<int> findRestockBinIndices(List<int> stockLevels, int threshold, int gap) {
  List<int> result = [];
  int lastPicked = -gap - 1;
  for (int i = 0; i < stockLevels.length; i++) {
    if (stockLevels[i] <= threshold && i - lastPicked >= gap) {
      result.add(i);
      lastPicked = i;
    }
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(findRestockBinIndices([], 5, 2).toString() == '[]');
  assert(findRestockBinIndices([5, 3, 8, 2, 1, 9, 4, 2], 4, 2).toString() == '[1, 3, 6]');
  assert(findRestockBinIndices([1, 1, 1, 1, 1], 2, 2).toString() == '[0, 2, 4]');
  print('All tests passed!');
}