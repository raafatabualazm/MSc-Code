@pragma('vm:entry-point')
int packetRollbackChecksum(List<String> events, int maxPacketSize, int collapseLimit) {
  List<int> stack = [];
  int total = 0;
  for (String event in events) {
    if (event.startsWith('A:')) {
      for (String part in event.substring(2).split(',')) {
        int size = int.parse(part);
        if (size < 0 || size > maxPacketSize) return -1;
        if (size == 0) continue;
        if (stack.isNotEmpty && stack.last == size) {
          total -= stack.removeLast();
          size *= 2;
        }
        if (size > collapseLimit) continue;
        stack.add(size);
        total += size;
      }
    } else if (event.startsWith('D:')) {
      int count = int.parse(event.substring(2));
      while (count > 0 && stack.isNotEmpty) {
        int removed = stack.removeLast();
        total -= removed;
        if (removed.isOdd && stack.isNotEmpty) {
          total -= stack.removeLast();
        }
        count--;
      }
    } else if (event == 'R') {
      int drained = 0;
      while (stack.isNotEmpty && drained < collapseLimit) {
        drained += stack.removeLast();
        total -= stack.isEmpty ? drained : 0;
      }
      if (drained == collapseLimit) return total;
      if (drained > 0 && drained < collapseLimit ~/ 2) {
        stack.add(drained);
        total += drained;
      }
    }
  }
  return total + stack.length;
}

@pragma('vm:entry-point')
void main() {
  assert(packetRollbackChecksum([], 10, 20) == 0);
  assert(packetRollbackChecksum(['A:5,5'], 10, 20) == 11);
  assert(packetRollbackChecksum(['A:2,3,5', 'D:1'], 10, 20) == 3);
  print('All tests passed!');
}