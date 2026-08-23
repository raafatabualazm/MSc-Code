@pragma('vm:entry-point')
List<List<int>> decodeAndCompressInventory(String encoded) {
  if (encoded.isEmpty) return [];
  List<List<int>> result = [];
  int prev = encoded[0].codeUnitAt(0) == 65 ? 90 : encoded[0].codeUnitAt(0) - 1;
  int count = 1;
  for (int i = 1; i < encoded.length; i++) {
    int current = encoded[i].codeUnitAt(0) == 65 ? 90 : encoded[i].codeUnitAt(0) - 1;
    if (current == prev) {
      count++;
    } else {
      result.add([count, prev]);
      prev = current;
      count = 1;
    }
  }
  result.add([count, prev]);
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(decodeAndCompressInventory("BBB").toString() == "[[3, 65]]");
  assert(decodeAndCompressInventory("AB").toString() == "[[1, 90], [1, 65]]");
  print('All tests passed!');
}