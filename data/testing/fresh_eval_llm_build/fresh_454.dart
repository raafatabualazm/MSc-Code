@pragma('vm:entry-point')
double calculateTideLedgerDepth(List<String> records) {
  List<double> stack = [];
  List<List<double>> history = [];
  for (String record in records) {
    if (record.startsWith('R')) {
      history.add(List<double>.from(stack));
      stack.add(double.parse(record.substring(1)));
    } else if (record == 'S') {
      if (stack.isNotEmpty) {
        history.add(List<double>.from(stack));
        double top = stack.removeLast();
        stack.add(top >= 0 ? top - 0.5 : top + 0.5);
      }
    } else if (record == 'M') {
      if (stack.length >= 2) {
        history.add(List<double>.from(stack));
        double a = stack.removeLast();
        double b = stack.removeLast();
        stack.add((a >= 0) == (b >= 0) ? (a + b) / 2.0 : a - b);
      }
    } else if (record == 'U' && history.isNotEmpty) {
      stack = history.removeLast();
    }
  }
  double total = 0.0;
  for (double value in stack) {
    total += value;
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(calculateTideLedgerDepth([]) == 0.0);
  assert(calculateTideLedgerDepth(['R1.0','R2.0','M']) == 1.5);
  assert(calculateTideLedgerDepth(['R-1.0','S']) == -0.5);
  print('All tests passed!');
}