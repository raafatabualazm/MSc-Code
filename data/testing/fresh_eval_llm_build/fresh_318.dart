@pragma('vm:entry-point')
Map<String, int> countFailingEndpointMentions(String logs) {
  final counts = <String, int>{};
  for (final line in logs.split('\n')) {
    final parts = line.split('|');
    if (parts.length == 3 && int.parse(parts[1]) >= 500) {
      counts[parts[2]] = (counts[parts[2]] ?? 0) + 1;
    }
  }
  return counts;
}

@pragma('vm:entry-point')
void main() {
  assert(countFailingEndpointMentions('').toString() == '{}');
  assert(countFailingEndpointMentions('a|500|/x').toString() == '{/x: 1}');
  assert(countFailingEndpointMentions('a|499|/x\nb|500|/x').length == 1);
  print('All tests passed!');
}