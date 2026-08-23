@pragma('vm:entry-point')
int countNonOverlappingInventoryItems(List<List<int>> items) {
  int count = 0;
  int n = items.length;
  for (int i = 0; i < n; i++) {
    bool overlaps = false;
    for (int j = 0; j < n; j++) {
      if (i == j) continue;
      var a = items[i];
      var b = items[j];
      if (!(a[2] < b[0] || b[2] < a[0] || a[3] < b[1] || b[3] < a[1])) {
        overlaps = true;
        break;
      }
    }
    if (!overlaps) count++;
  }
  return count;
}

@pragma('vm:entry-point')
void main() {
  assert(countNonOverlappingInventoryItems([]) == 0);
  assert(countNonOverlappingInventoryItems([[0,0,2,2]]) == 1);
  assert(countNonOverlappingInventoryItems([[0,0,2,2], [3,3,5,5]]) == 2);
  print('All tests passed!');
}