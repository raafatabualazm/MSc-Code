@pragma('vm:entry-point')
String? replayMorseUndoQueue(List<String> events, int maxRun) {
  final queue = <String>[];
  var current = '';
  for (final event in events) {
    if (event == '.' || event == '-') {
      current += event;
    } else if (event == '/') {
      if (current.isEmpty) continue;
      var run = 1;
      for (var i = 1; i < current.length; i++) {
        run = current[i] == current[i - 1] ? run + 1 : 1;
        if (run > maxRun) return null;
      }
      for (final saved in queue) {
        if (saved.length != current.length) continue;
        var reversed = true;
        for (var j = 0; j < saved.length; j++) {
          if (saved[j] != current[current.length - 1 - j]) {
            reversed = false;
            break;
          }
        }
        if (reversed) return null;
      }
      queue.add(current);
      current = '';
    } else if (event == '<') {
      if (current.isNotEmpty) {
        current = current.substring(0, current.length - 1);
      } else if (queue.isNotEmpty) {
        current = queue.removeLast();
        if (current.isEmpty) return null;
        current = current.substring(0, current.length - 1);
      } else {
        return null;
      }
    } else {
      return null;
    }
  }
  if (current.isNotEmpty) queue.add(current);
  return queue.join(' ');
}

@pragma('vm:entry-point')
void main() {
  assert(replayMorseUndoQueue([], 2) == '');
  assert(replayMorseUndoQueue(['.', '-', '/'], 2) == '.-');
  assert(replayMorseUndoQueue(['.', '.', '.', '/'], 2) == null);
  print('All tests passed!');
}