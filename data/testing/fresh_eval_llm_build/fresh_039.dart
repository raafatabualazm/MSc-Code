@pragma('vm:entry-point')
List<int> reconcileQrModuleQueue(List<int> signals) {
  List<int> queue = [];
  List<List<int>> history = [];
  for (int signal in signals) {
    if (signal > 0) {
      if (queue.isNotEmpty && queue.last == signal) continue;
      history.add(List<int>.from(queue));
      queue.add(signal);
    } else if (signal == 0) {
      if (queue.length < 2) continue;
      history.add(List<int>.from(queue));
      int first = queue.removeAt(0);
      queue.add(first);
    } else if (signal == -1) {
      if (history.isEmpty) return queue;
      queue = List<int>.from(history.removeLast());
    } else {
      int threshold = -signal;
      history.add(List<int>.from(queue));
      int removed = 0;
      while (queue.isNotEmpty && queue.first < threshold) {
        removed += queue.removeAt(0);
        if (queue.isNotEmpty && queue.first == removed) {
          removed += queue.removeAt(0);
        }
      }
      if (removed > 0) {
        if (queue.isNotEmpty && queue.last > removed) {
          queue.insert(0, removed);
        } else {
          queue.add(removed);
        }
      }
    }
  }
  return queue;
}

@pragma('vm:entry-point')
void main() {
  assert(reconcileQrModuleQueue([1, 2, 0]).toString() == '[2, 1]');
  assert(reconcileQrModuleQueue([2, 3, 4, -4]).length == 2);
  assert(reconcileQrModuleQueue([-1, 5]).toString() == '[]');
  print('All tests passed!');
}