@pragma('vm:entry-point')
int calculateWifiBinPressure(List<String> events) {
  List<int> bins = [];
  List<List<int>> history = [];
  for (final event in events) {
    if (event == 'undo') {
      if (history.isNotEmpty) bins = List<int>.from(history.removeLast());
    } else {
      history.add(List<int>.from(bins));
      if (event == 'weak' || event == 'mid' || event == 'strong') {
        bins.add(event == 'weak' ? 1 : event == 'mid' ? 2 : 3);
      } else if (event == 'merge' && bins.length >= 2) {
        int a = bins.removeLast(), b = bins.removeLast();
        bins.add(a > b ? a - b + 1 : b - a + 1);
      } else if (event == 'fade' && bins.isNotEmpty) {
        int next = bins.removeLast() - 1;
        if (next > 0) bins.add(next);
      }
    }
  }
  int total = 0;
  for (final v in bins) {
    total += v * bins.length;
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(calculateWifiBinPressure([]) == 0);
  assert(calculateWifiBinPressure(['weak', 'mid', 'merge']) == 2);
  assert(calculateWifiBinPressure(['strong', 'fade', 'undo']) == 3);
  print('All tests passed!');
}