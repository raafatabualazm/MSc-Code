@pragma('vm:entry-point')
String findBarByPosition(List<int> scanData) {
  if (scanData.isEmpty) return "Empty";
  int x = scanData[0];
  if (x <= 0) return "Invalid";
  int n = scanData.length - 1;
  if (n == 0) return "Empty";
  int total = 0;
  for (int i = 1; i < scanData.length; i++) {
    total += scanData[i];
  }
  if (x > total) return "Out of range";
  int low = 0, high = n - 1;
  while (low <= high) {
    int mid = (low + high) ~/ 2;
    int sum = 0;
    for (int i = 1; i <= mid + 1; i++) {
      sum += scanData[i];
    }
    if (sum > x) {
      if (sum - scanData[mid + 1] > x) {
        high = mid - 1;
      } else {
        String color = (mid % 2 == 0) ? "B" : "W";
        return "$color$mid";
      }
    } else {
      low = mid + 1;
    }
  }
  return "Out of range";
}

@pragma('vm:entry-point')
void main() {
  assert(findBarByPosition([]) == "Empty");
  assert(findBarByPosition([5, 10, 20]) == "B0");
  assert(findBarByPosition([10, 10, 20]) == "W1");
  print('All tests passed!');
}