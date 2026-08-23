@pragma('vm:entry-point')
int qrModuleUndoChecksum(String log, int limit) {
  List<int> applied = [];
  List<int> redo = [];
  int current = 0;
  int checksum = 0;
  for (int i = 0; i < log.length; i++) {
    String c = log[i];
    if (c == '#') {
      checksum += current <= limit ? current : limit - current;
      continue;
    }
    if (c == 'U') {
      if (applied.isNotEmpty) {
        int delta = applied.removeLast();
        current -= delta;
        redo.add(delta);
      }
      continue;
    }
    if (c == 'R') {
      if (redo.isNotEmpty) {
        int delta = redo.removeAt(0);
        current += delta;
        applied.add(delta);
      }
    } else if (c.codeUnitAt(0) >= 49 && c.codeUnitAt(0) <= 57) {
      if (applied.isEmpty) continue;
      int delta = applied.last;
      for (int j = 0; j < c.codeUnitAt(0) - 48; j++) {
        current += delta;
        if (current < 0 || current > limit * 2) return -1;
        applied.add(delta);
      }
      redo.clear();
    } else {
      int delta = c == 'B' ? 1 : (c == 'W' ? -1 : 0);
      if (delta == 0) continue;
      current += delta;
      if (current < 0 || current > limit * 2) return -1;
      applied.add(delta);
      redo.clear();
    }
  }
  return checksum + current;
}

@pragma('vm:entry-point')
void main() {
  assert(qrModuleUndoChecksum('', 3) == 0);
  assert(qrModuleUndoChecksum('BB#U#', 2) == 4);
  assert(qrModuleUndoChecksum('B5', 2) == -1);
  print('All tests passed!');
}