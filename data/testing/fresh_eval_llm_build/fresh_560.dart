@pragma('vm:entry-point')
List<int> decodePhasePulseLedger(String plan) {
  if (plan.isEmpty) return [];
  List<int> result = [];
  for (String block in plan.split('|')) {
    int score = 0;
    int expanded = 0;
    String previous = '';
    for (int i = 0; i < block.length; i++) {
      String ch = block[i];
      if (ch == ',') {
        previous = '';
        continue;
      }
      if (ch != 'R' && ch != 'Y' && ch != 'G') return [];
      int count = 0;
      int j = i + 1;
      while (j < block.length) {
        int code = block.codeUnitAt(j);
        if (code < 48 || code > 57) break;
        count = count * 10 + code - 48;
        j++;
      }
      count = count == 0 ? 1 : count;
      for (int k = 0; k < count; k++) {
        score += ch == 'R' ? 3 : (ch == 'Y' ? 2 : 1);
        if (previous == ch) score--;
        previous = ch;
        expanded++;
      }
      i = j - 1;
    }
    result.add(score + expanded);
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(decodePhasePulseLedger('').toString() == '[]');
  assert(decodePhasePulseLedger('R2|G3').toString() == '[7, 4]');
  assert(decodePhasePulseLedger('G2,G2').toString() == '[6]');
  print('All tests passed!');
}