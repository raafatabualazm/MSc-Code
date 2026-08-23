@pragma('vm:entry-point')
String elevatorServiceOrder(List<int> floors, int current) {
  var sorted = List<int>.from(floors);
  sorted.sort((a, b) {
    if (a >= current && b >= current) return a - b;
    if (a < current && b < current) return b - a;
    return a >= current ? -1 : 1;
  });
  return sorted.join(',');
}

@pragma('vm:entry-point')
void main() {
  assert(elevatorServiceOrder([], 0) == "");
  assert(elevatorServiceOrder([5], 5) == "5");
  assert(elevatorServiceOrder([1,5,3], 3) == "3,5,1");
  print('All tests passed!');
}