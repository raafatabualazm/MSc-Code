@pragma('vm:entry-point')
List<String> roundedCycleClosings(String dailyNetCents, int cycleDays) {
  List<String> parts = dailyNetCents.isEmpty ? [] : dailyNetCents.split(',');
  List<String> out = [];
  int sum = 0, start = 0;
  for (int i = 0; i < parts.length; i++) {
    sum += int.parse(parts[i]);
    if (i - start + 1 == cycleDays || i == parts.length - 1) {
      int rounded = ((sum + (sum >= 0 ? 5 : -5)) ~/ 10) * 10;
      out.add('$start-$i:$rounded');
      sum = 0;
      start = i + 1;
    }
  }
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(roundedCycleClosings('', 3).toString() == '[]');
  assert(roundedCycleClosings('4,5', 2).toString() == '[0-1:10]');
  assert(roundedCycleClosings('8,8,8', 2).toString() == '[0-1:20, 2-2:10]');
  print('All tests passed!');
}