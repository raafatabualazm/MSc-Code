@pragma('vm:entry-point')
List<int> filterBrightPixelWeekdays(List<int> data) {
  if (data.length < 3) return [];
  int weekday = data[0];
  int threshold = data[1];
  List<int> result = [];
  for (int i = 2; i < data.length; i++) {
    if (data[i] > threshold) {
      result.add(weekday);
    }
    weekday = (weekday + 1) % 7;
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(filterBrightPixelWeekdays([0, 100, 101]).toString() == '[0]');
  assert(filterBrightPixelWeekdays([6, 100, 101, 102]).toString() == '[6, 0]');
  assert(filterBrightPixelWeekdays([0, 100, 99]).length == 0);
  print('All tests passed!');
}