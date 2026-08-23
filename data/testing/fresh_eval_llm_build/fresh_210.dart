@pragma('vm:entry-point')
int rowContainingKthVisibleChar(List<String> rows, int k) {
  List<int> cum = [];
  int total = 0;
  for (var row in rows) {
    for (int i = 0; i < row.length; i++) if (row[i] != ' ') total++;
    cum.add(total);
  }
  if (k <= 0 || k > total) return -1;
  int lo = 0, hi = rows.length - 1, ans = -1;
  while (lo <= hi) {
    int mid = (lo + hi) ~/ 2;
    if (cum[mid] >= k) {
      ans = mid;
      hi = mid - 1;
    } else {
      lo = mid + 1;
    }
  }
  return ans;
}

@pragma('vm:entry-point')
void main() {
  assert(rowContainingKthVisibleChar(["X"], 1) == 0);
  assert(rowContainingKthVisibleChar(["X", "Y"], 2) == 1);
  assert(rowContainingKthVisibleChar([" "], 1) == -1);
  print('All tests passed!');
}