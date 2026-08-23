@pragma('vm:entry-point')
String recentTrafficPhaseTape(String events, int keep) {
  var stack = <String>[];
  for (var i = 0; i < events.length; i++) {
    var c = events[i];
    if (c == '!') {
      if (stack.isNotEmpty) stack.removeLast();
    } else {
      stack.add(c);
    }
  }
  return stack.skip(stack.length > keep ? stack.length - keep : 0).join();
}

@pragma('vm:entry-point')
void main() {
  assert(recentTrafficPhaseTape("GRY", 2) == "RY");
  assert(recentTrafficPhaseTape("G!R", 5) == "R");
  assert(recentTrafficPhaseTape("!!", 1) == "");
  print('All tests passed!');
}