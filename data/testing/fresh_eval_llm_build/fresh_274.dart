@pragma('vm:entry-point')
String resolveDnaNoiseTrail(String trail) {
  var stack = <String>[];
  for (var ch in trail.split('')) {
    if (ch == 'N') {
      if (stack.isNotEmpty) stack.removeLast();
    } else if ('ATCG'.contains(ch)) {
      stack.add(ch);
    }
  }
  return stack.join();
}

@pragma('vm:entry-point')
void main() {
  assert(resolveDnaNoiseTrail('ATNCG') == 'ACG');
  assert(resolveDnaNoiseTrail('NNAC') == 'AC');
  assert(resolveDnaNoiseTrail('ACGTNN') == 'AC');
  print('All tests passed!');
}