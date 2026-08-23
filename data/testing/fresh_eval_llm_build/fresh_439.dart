@pragma('vm:entry-point')
List<int> arrangeElevatorStopsByLobbyGap(List<int> requests) {
  var ordered = List<int>.from(requests);
  ordered.sort((a, b) {
    var da = a.abs(), db = b.abs();
    if (da != db) return da.compareTo(db);
    if ((a >= 0) != (b >= 0)) return b.compareTo(a);
    return a.compareTo(b);
  });
  return ordered;
}

@pragma('vm:entry-point')
void main() {
  assert(arrangeElevatorStopsByLobbyGap([3, -1, 1, -3]).toString() == '[1, -1, 3, -3]');
  assert(arrangeElevatorStopsByLobbyGap([]).toString() == '[]');
  assert(arrangeElevatorStopsByLobbyGap([0, -2, 2]).toString() == '[0, 2, -2]');
  print('All tests passed!');
}