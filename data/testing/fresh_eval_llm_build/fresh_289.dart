@pragma('vm:entry-point')
bool hasWholeIngredientRatios(List<double> amounts) {
  if (amounts.isEmpty) return false;
  double minPos = double.infinity;
  for (double a in amounts) {
    if (a > 0 && a < minPos) minPos = a;
  }
  if (minPos == double.infinity) return false;
  for (double a in amounts) {
    double ratio = a / minPos;
    if (ratio != ratio.roundToDouble()) return false;
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(hasWholeIngredientRatios([2.0, 4.0, 8.0]) == true);
  assert(hasWholeIngredientRatios([1.0, 2.5]) == false);
  assert(hasWholeIngredientRatios([]) == false);
  print('All tests passed!');
}