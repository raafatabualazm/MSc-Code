@pragma('vm:entry-point')
int totalTrainDwellMinutes(List<String> timetable) {
  List<int> stack = [];
  int total = 0;
  int lastDeparture = -1;
  for (var op in timetable) {
    if (op.startsWith('A')) {
      stack.add(int.parse(op.substring(2)));
    } else if (op.startsWith('D')) {
      int time = int.parse(op.substring(2));
      if (stack.isEmpty || time <= stack.last || time <= lastDeparture) return -1;
      total += time - stack.last;
      lastDeparture = time;
      stack.removeLast();
    }
  }
  if (stack.isNotEmpty) return -1;
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(totalTrainDwellMinutes([]) == 0);
  assert(totalTrainDwellMinutes(['A,100', 'D,150']) == 50);
  assert(totalTrainDwellMinutes(['A,100']) == -1);
  print('All tests passed!');
}