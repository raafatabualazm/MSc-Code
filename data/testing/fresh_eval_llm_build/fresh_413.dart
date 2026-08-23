@pragma('vm:entry-point')
int finalTelemetrySum(List<int> actions) {
  List<int> main = [], redo = [];
  int sum = 0;
  for (int a in actions) {
    if (a > 0) { main.add(a); redo.clear(); sum += a; }
    else if (a < 0) {
      for (int i = 0, u = -a; i < u && main.isNotEmpty; i++) { int r = main.removeLast(); redo.add(r); sum -= r; }
    } else if (redo.isNotEmpty) { int r = redo.removeLast(); main.add(r); sum += r; }
  }
  return sum;
}

@pragma('vm:entry-point')
void main() {
  assert(finalTelemetrySum([]) == 0);
  assert(finalTelemetrySum([5]) == 5);
  assert(finalTelemetrySum([3, 5, -1, 0]) == 8);
  print('All tests passed!');
}