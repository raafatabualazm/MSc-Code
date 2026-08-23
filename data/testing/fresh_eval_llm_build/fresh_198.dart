@pragma('vm:entry-point')
List<int> warehouseRestockSignals(List<int> slots) {
  List<int> result = [];
  for (int code in slots) {
    int qty = code & 15;
    int flags = (code >> 4) & 15;
    int active = 0;
    for (int b = 0; b < 4; b++) {
      if ((flags & (1 << b)) != 0) active++;
    }
    int rotated = ((qty << 1) & 15) | (qty >> 3);
    if ((flags & 4) != 0) {
      if (qty <= 1) {
        result.add(rotated + active + 6);
      } else {
        result.add(rotated - active);
      }
    } else if ((flags & 3) == 3) {
      result.add(rotated + qty);
    } else {
      result.add(rotated ^ active);
    }
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(warehouseRestockSignals([]).toString() == '[]');
  assert(warehouseRestockSignals([64, 66]).toString() == '[7, 3]');
  assert(warehouseRestockSignals([31, 63]).toString() == '[14, 30]');
  print('All tests passed!');
}