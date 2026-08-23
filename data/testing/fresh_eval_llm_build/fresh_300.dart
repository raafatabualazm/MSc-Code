@pragma('vm:entry-point')
List<int> rankChessSquaresByCenter(List<int> squares) {
  var result = List<int>.from(squares);
  int score(int s) => ((s ~/ 8) - 3).abs() + ((s % 8) - 3).abs();
  result.sort((a, b) {
    int d = score(a) - score(b);
    int e = ((a ~/ 8) - (a % 8)).abs() - ((b ~/ 8) - (b % 8)).abs();
    return d != 0 ? d : (e != 0 ? e : a - b);
  });
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(rankChessSquaresByCenter([0, 27, 63]).toString() == '[27, 0, 63]');
  assert(rankChessSquaresByCenter([]).toString() == '[]');
  assert(rankChessSquaresByCenter([18, 20, 9]).toString() == '[18, 20, 9]');
  print('All tests passed!');
}