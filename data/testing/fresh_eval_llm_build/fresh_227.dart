@pragma('vm:entry-point')
List<int> findDarkRunStartIndices(List<int> modules, int k) {
  final List<int> result = [];
  final int n = modules.length;
  int i = 0;
  while (i < n) {
    if (modules[i] == 1) {
      int start = i;
      int runLen = 0;
      while (i < n && modules[i] == 1) {
        runLen++;
        i++;
      }
      if (runLen == k) {
        result.add(start);
      }
    } else {
      i++;
    }
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(findDarkRunStartIndices([1,1,0,1,1,1,0,1,1], 2).toString() == '[0, 7]');
  assert(findDarkRunStartIndices([1,1,1], 3).toString() == '[0]');
  assert(findDarkRunStartIndices([], 1).toString() == '[]');
  print('All tests passed!');
}