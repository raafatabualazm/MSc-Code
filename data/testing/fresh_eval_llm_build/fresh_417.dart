@pragma('vm:entry-point')
double sortedServerLogEdgeSpan(List<String> logs) {
  if (logs.length < 2) return 0.0;
  var sorted = [...logs]
    ..sort((a, b) {
      int da = RegExp(r'\d').allMatches(a).length;
      int db = RegExp(r'\d').allMatches(b).length;
      return da != db ? da - db : b.compareTo(a);
    });
  return (sorted.first.length - sorted.last.length).abs() / 4.0;
}

@pragma('vm:entry-point')
void main() {
  assert(sortedServerLogEdgeSpan([]) == 0.0);
  assert(sortedServerLogEdgeSpan(['a1', 'bbb22']) == 0.75);
  assert(sortedServerLogEdgeSpan(['log', 'a1']) == 0.25);
  print('All tests passed!');
}