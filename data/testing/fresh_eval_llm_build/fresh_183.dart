@pragma('vm:entry-point')
String totalManhattanDistance(List<String> squares) {
  if (squares.length <= 1) return "0";
  int total = 0;
  for (int i = 1; i < squares.length; i++) {
    var a = squares[i-1];
    var b = squares[i];
    total += (a.codeUnitAt(0) - b.codeUnitAt(0)).abs() + (int.parse(a[1]) - int.parse(b[1])).abs();
  }
  return total.toString();
}

@pragma('vm:entry-point')
void main() {
  assert(totalManhattanDistance([]) == "0");
  assert(totalManhattanDistance(["a1"]) == "0");
  assert(totalManhattanDistance(["a1", "b2"]) == "2");
  print('All tests passed!');
}