@pragma('vm:entry-point')
int theaterChecksumDecoded(String encoded) {
  if (encoded.isEmpty) return 0;
  final rows = encoded.split('|');
  int total = 0;
  for (int r = 0; r < rows.length; r++) {
    final row = rows[r];
    if (row.isEmpty) return 0;
    int occupied = 0;
    int seats = 0;
    int i = 0;
    while (i < row.length) {
      int numStart = i;
      while (i < row.length && row[i].compareTo('0') >= 0 && row[i].compareTo('9') <= 0) i++;
      if (i >= row.length || i == numStart) return 0;
      final count = int.parse(row.substring(numStart, i));
      final type = row[i];
      i++;
      seats += count;
      if (type == 'O') occupied += count;
      else if (type != 'E') return 0;
    }
    if (seats == 0) return 0;
    if (occupied * 2 > seats) {
      total -= (r + 1) * occupied;
    } else {
      total += (r + 1) * occupied;
    }
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(theaterChecksumDecoded("2O2E|1O3E") == 4);
  assert(theaterChecksumDecoded("") == 0);
  assert(theaterChecksumDecoded("6O4E|4O6E") == 2);
  print('All tests passed!');
}