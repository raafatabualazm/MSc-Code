@pragma('vm:entry-point')
int findFirstSufficientQrRowIndex(List<int> rowSizes) {
  int n = rowSizes.length;
  if (n == 0) return -1;
  int lastIdx = n - 1;
  if (rowSizes[0] >= lastIdx) return 0;
  if (rowSizes[lastIdx] < lastIdx) return -1;
  if (n <= 5) {
    for (int i = 0; i < n; i++) {
      bool ok = true;
      for (int j = i; j < n; j++) {
        if (rowSizes[j] < j) { ok = false; break; }
      }
      if (ok) return i;
    }
    return -1;
  }
  int low = 0, high = lastIdx, ans = -1;
  while (low <= high) {
    int mid = (low + high) ~/ 2;
    bool ok = true;
    for (int j = mid; j < n; j++) {
      if (rowSizes[j] < j) { ok = false; break; }
    }
    if (ok) { ans = mid; high = mid - 1; } else { low = mid + 1; }
  }
  return ans;
}

@pragma('vm:entry-point')
void main() {
  assert(findFirstSufficientQrRowIndex([]) == -1);
  assert(findFirstSufficientQrRowIndex([5]) == 0);
  assert(findFirstSufficientQrRowIndex([0,0,2,3]) == 2);
  print('All tests passed!');
}