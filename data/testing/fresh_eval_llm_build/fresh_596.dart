@pragma('vm:entry-point')
String scaleIngredientLedger(String ledger, int multiplier) {
  if (ledger.trim().isEmpty || multiplier == 0) {
    return '';
  }
  List<String> output = [];
  for (String raw in ledger.split('|')) {
    String item = raw.trim();
    if (item.isEmpty) {
      continue;
    }
    List<String> parts = item.split(':');
    if (parts.length != 2) {
      continue;
    }
    String qty = parts[0].trim();
    String name = parts[1].trim();
    int slashCount = 0;
    for (int i = 0; i < qty.length; i++) {
      if (qty[i] == '/') {
        slashCount++;
      } else if ('0123456789'.indexOf(qty[i]) == -1) {
        slashCount = 99;
        break;
      }
    }
    if (slashCount > 1 || name.isEmpty) {
      continue;
    }
    if (slashCount == 0) {
      int value = int.parse(qty) * multiplier;
      if (value <= 0) {
        continue;
      }
      output.add('$value $name');
    } else {
      List<String> frac = qty.split('/');
      int top = int.parse(frac[0]) * multiplier;
      int bottom = int.parse(frac[1]);
      if (top <= 0 || bottom == 0) {
        continue;
      }
      while (top % 2 == 0 && bottom % 2 == 0) {
        top ~/= 2;
        bottom ~/= 2;
      }
      output.add('$top/$bottom $name');
    }
  }
  return output.join(', ');
}

@pragma('vm:entry-point')
void main() {
  assert(scaleIngredientLedger('2:flour|1/2:milk', 2) == '4 flour, 1/1 milk');
  assert(scaleIngredientLedger('', 3) == '');
  assert(scaleIngredientLedger('5:salt|x:pepper', 1) == '5 salt');
  print('All tests passed!');
}