@pragma('vm:entry-point')
int consolidateAndDiscountInventory(List<int> quantities, int mergeThreshold) {
  if (quantities.isEmpty) return 0;
  if (mergeThreshold < 0) {
    return quantities.fold(0, (sum, e) => sum + e);
  }
  List<int> bins = List<int>.from(quantities);
  bool changed = true;
  while (changed) {
    changed = false;
    int i = 0;
    while (i < bins.length - 1) {
      if ((bins[i] - bins[i + 1]).abs() <= mergeThreshold) {
        int merged = ((bins[i] + bins[i + 1]) * 9) ~/ 10;
        bins.removeAt(i + 1);
        bins[i] = merged;
        changed = true;
        i++;
      } else {
        i++;
      }
    } 
  }
  int total = 0;
  for (int q in bins) {
    if (q > 100) {
      total += (q * 9) ~/ 10;
    } else {
      total += q;
    }
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(consolidateAndDiscountInventory([], 0) == 0);
  assert(consolidateAndDiscountInventory([100, 200], 0) == 280);
  assert(consolidateAndDiscountInventory([5, 5, 5], 5) == 12);
  print('All tests passed!');
}