@pragma('vm:entry-point')
bool isMedianBlackModuleInEvenRow(List<int> cumulativeBlackModulesPerRow) {
  if (cumulativeBlackModulesPerRow.isEmpty || cumulativeBlackModulesPerRow.last == 0) return false;
  int target = cumulativeBlackModulesPerRow.last ~/ 2;
  int left = 0, right = cumulativeBlackModulesPerRow.length;
  while (left < right) {
    int mid = (left + right) ~/ 2;
    if (cumulativeBlackModulesPerRow[mid] <= target) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left % 2 == 0;
}

@pragma('vm:entry-point')
void main() {
  assert(isMedianBlackModuleInEvenRow([2, 5]) == false);
  assert(isMedianBlackModuleInEvenRow([1]) == true);
  assert(isMedianBlackModuleInEvenRow([]) == false);
  print('All tests passed!');
}