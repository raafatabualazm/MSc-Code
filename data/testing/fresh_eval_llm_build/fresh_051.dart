@pragma('vm:entry-point')
double computeShippedPalletAverage(List<String> commands, double capacityLimit, int maxStackDepth) {
  List<double> stack = [];
  List<double> staging = [];
  List<double> shipped = [];

  for (int i = 0; i < commands.length; i++) {
    final cmd = commands[i];
    if (cmd == 'POP') {
      if (stack.isEmpty) continue;
      final top = stack.removeLast();
      staging.add(top);
    } else if (cmd == 'SHIP') {
      if (staging.isEmpty) continue;
      double stagingSum = 0.0;
      for (int j = 0; j < staging.length; j++) {
        stagingSum += staging[j];
      }
      if (stagingSum <= capacityLimit) {
        for (int j = 0; j < staging.length; j++) {
          shipped.add(staging[j]);
        }
      }
      staging = [];
    } else {
      final weight = double.tryParse(cmd);
      if (weight == null || weight <= 0.0) continue;
      if (stack.length >= maxStackDepth) continue;
      stack.add(weight);
    }
  }
  if (shipped.isEmpty) return 0.0;
  double total = 0.0;
  for (int i = 0; i < shipped.length; i++) {
    total += shipped[i];
  }
  return total / shipped.length;
}

@pragma('vm:entry-point')
void main() {
  assert(computeShippedPalletAverage(['4.0','2.0','POP','POP','SHIP'], 10.0, 5) == 3.0);
  assert(computeShippedPalletAverage(['8.0','POP','SHIP','2.0','POP','SHIP'], 6.0, 3) == 2.0);
  assert(computeShippedPalletAverage([], 100.0, 5) == 0.0);
  print('All tests passed!');
}