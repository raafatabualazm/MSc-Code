@pragma('vm:entry-point')
List<int> scoreMazeCellBands(String mapData) {
  if (mapData.isEmpty) return [];
  List<int> result = [];
  List<String> rows = mapData.split(';');
  for (String row in rows) {
    int score = 0;
    int last = 0;
    bool touched = false;
    List<String> cells = row.split(',');
    for (int i = 0; i < cells.length; i++) {
      String cell = cells[i].trim();
      if (cell.isEmpty || cell == '.') continue;
      touched = true;
      String kind = cell[0];
      int value = cell.length == 1 ? 0 : int.tryParse(cell.substring(1)) ?? -99;
      if (value == -99) {
        score -= cell.length;
        continue;
      }
      if (kind == 'X' && value < 0) break;
      if (kind == 'G') {
        score += value < 0 ? -value : value;
      } else if (kind == 'T') {
        score += value.isEven ? value ~/ 2 : value * 2;
      } else if (kind == 'R') {
        score += last + value;
      } else {
        score += value - i;
      }
      last = value;
    }
    result.add(touched ? score : 0);
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(scoreMazeCellBands('').toString() == '[]');
  assert(scoreMazeCellBands('G3,T4,R2').toString() == '[11]');
  assert(scoreMazeCellBands('X-1,G9').toString() == '[0]');
  print('All tests passed!');
}