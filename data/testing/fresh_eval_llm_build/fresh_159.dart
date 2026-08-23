@pragma('vm:entry-point')
List<List<int>> segmentFloorRequestsByProximity(List<int> floors) {
  if (floors.isEmpty) return [];
  List<List<int>> result = [];
  List<int> current = [floors[0]];
  for (int i = 1; i < floors.length; i++) {
    if ((floors[i] - floors[i - 1]).abs() > 3) {
      List<int> seg = current.toSet().toList()..sort();
      result.add(seg);
      current = [floors[i]];
    } else {
      current.add(floors[i]);
    }
  }
  List<int> seg = current.toSet().toList()..sort();
  result.add(seg);
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(segmentFloorRequestsByProximity([]).toString() == '[]');
  assert(segmentFloorRequestsByProximity([1, 2, 6, 7]).toString() == '[[1, 2], [6, 7]]');
  assert(segmentFloorRequestsByProximity([3, 3, 4]).toString() == '[[3, 4]]');
  print('All tests passed!');
}