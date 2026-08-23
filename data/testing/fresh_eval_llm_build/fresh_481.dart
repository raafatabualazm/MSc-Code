@pragma('vm:entry-point')
bool validatesElevatorUndoLog(List<String> log) {
  List<int> pending = [];
  bool servedAny = false;
  for (String entry in log) {
    if (entry.startsWith('+')) {
      int floor = int.parse(entry.substring(1));
      if (floor == 0 || floor.abs() > 50) return false;
      if (pending.isNotEmpty && pending.last == floor) return false;
      pending.add(floor);
      continue;
    }
    if (entry == 'UNDO') {
      if (pending.isEmpty) return false;
      pending.removeLast();
      continue;
    }
    if (!entry.startsWith('SERVE:')) return false;
    String payload = entry.substring(6);
    if (payload.isEmpty) return false;
    servedAny = true;
    for (String piece in payload.split(',')) {
      if (piece.isEmpty) return false;
      int floor = int.parse(piece);
      if (pending.isEmpty || pending.last != floor) return false;
      pending.removeLast();
      if (pending.isNotEmpty) {
        for (int i = pending.length - 1; i > 0; i--) {
          if (pending[i] == pending[i - 1]) return false;
        }
      }
    }
  }
  return servedAny && pending.isEmpty;
}

@pragma('vm:entry-point')
void main() {
  assert(validatesElevatorUndoLog(['+3', '+5', 'SERVE:5,3']) == true);
  assert(validatesElevatorUndoLog(['UNDO']) == false);
  assert(validatesElevatorUndoLog(['+2', '+2']) == false);
  print('All tests passed!');
}