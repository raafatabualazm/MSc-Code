@pragma('vm:entry-point')
List<String> decodeChessSquareBursts(String encoded) {
  List<String> out = [];
  if (encoded.isEmpty) return out;
  List<String> parts = encoded.split(',');
  for (String part in parts) {
    if (part.length < 4) continue;
    int idx = 0;
    int count = 0;
    while (idx < part.length && part.codeUnitAt(idx) >= 48 && part.codeUnitAt(idx) <= 57) {
      count = count * 10 + part.codeUnitAt(idx) - 48;
      idx++;
    }
    if (count <= 0 || idx + 2 >= part.length) continue;
    String file = part[idx];
    int rank = int.tryParse(part[idx + 1]) ?? -1;
    String mark = part.substring(idx + 2);
    if (file.compareTo('a') < 0 || file.compareTo('h') > 0 || rank < 1 || rank > 8) {
      if (count > 8) return [];
      continue;
    }
    int check = ((file.codeUnitAt(0) - 96) * 2 + rank + count) % 26;
    if (mark != String.fromCharCode(65 + check)) continue;
    for (int i = 0; i < count; i++) {
      int shift = (i + rank).isEven ? 0 : 7 - (file.codeUnitAt(0) - 97) * 2;
      String nextFile = String.fromCharCode(file.codeUnitAt(0) + shift);
      out.add('$nextFile$rank');
    }
  }
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(decodeChessSquareBursts('').isEmpty);
  assert(decodeChessSquareBursts('2b3J').toString() == '[g3, b3]');
  assert(decodeChessSquareBursts('10a1N').length == 10);
  print('All tests passed!');
}