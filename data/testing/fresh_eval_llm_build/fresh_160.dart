@pragma('vm:entry-point')
List<int> parseInventoryQuantities(String inventory) {
  List<int> result = [];
  if (inventory.isEmpty) return result;
  List<String> entries = inventory.split(',');
  for (String entry in entries) {
    if (entry.isEmpty) continue;
    int multiplier = 1;
    String token = entry.trim();
    if (token.startsWith('!')) {
      multiplier = 3;
      token = token.substring(1);
    } else if (token.startsWith('*')) {
      multiplier = 2;
      token = token.substring(1);
    }
    int colonIdx = token.indexOf(':');
    if (colonIdx <= 0) continue;
    String name = token.substring(0, colonIdx);
    String quantStr = token.substring(colonIdx + 1);
    bool nameCorrupted = false;
    for (int i = 0; i < name.length; i++) {
      int code = name.codeUnitAt(i);
      if (code >= 48 && code <= 57) { nameCorrupted = true; break; }
    }
    if (nameCorrupted) continue;
    if (quantStr.isEmpty) continue;
    bool validQuant = true;
    int qty = 0;
    int start = 0;
    if (quantStr[0] == '-') { start = 1; }
    if (start >= quantStr.length) continue;
    for (int j = start; j < quantStr.length; j++) {
      int c = quantStr.codeUnitAt(j);
      if (c < 48 || c > 57) { validQuant = false; break; }
      qty = qty * 10 + (c - 48);
    }
    if (!validQuant) continue;
    if (quantStr[0] == '-') qty = -qty;
    result.add(qty * multiplier);
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(parseInventoryQuantities('').toString() == '[]');
  assert(parseInventoryQuantities('*sword:3,!shield:1,potion:5').toString() == '[6, 3, 5]');
  assert(parseInventoryQuantities('axe2:4,bow:2').toString() == '[2]');
  print('All tests passed!');
}