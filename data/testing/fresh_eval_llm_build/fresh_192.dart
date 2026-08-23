@pragma('vm:entry-point')
List<int> parseElevatorRequests(String requests) {
  final result = <int>[];
  for (final token in requests.split(',')) {
    final trimmed = token.trim();
    if (trimmed.isNotEmpty) {
      final match = RegExp(r'^(\d+)').firstMatch(trimmed);
      if (match != null) {
        final floor = int.parse(match.group(1)!);
        if (floor >= 1 && floor <= 15) result.add(floor);
      }
    }
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(parseElevatorRequests("").toString() == "[]");
  assert(parseElevatorRequests("5,10,15").toString() == "[5, 10, 15]");
  assert(parseElevatorRequests("0,1,16").toString() == "[1]");
  print('All tests passed!');
}