@pragma('vm:entry-point')
bool isValidEncodedTideReadings(String encoded) {
  if (encoded.isEmpty) return false;
  List<int> heights = [];
  for (int i = 0; i < encoded.length; i++) {
    int code = encoded.codeUnitAt(i) - 3;
    if (code < 48 || code > 57) return false;
    heights.add(code - 48);
  }
  int sum = 0;
  for (int i = 0; i < heights.length; i++) {
    sum += heights[i];
    if (i < heights.length - 1) {
      if ((heights[i] - heights[i+1]).abs() > 2) return false;
    }
  }
  if ((heights.first - heights.last).abs() > 1) return false;
  if (sum % 2 != 0) return false;
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(isValidEncodedTideReadings('') == false);
  assert(isValidEncodedTideReadings('3') == true);
  assert(isValidEncodedTideReadings('34') == false);
  print('All tests passed!');
}