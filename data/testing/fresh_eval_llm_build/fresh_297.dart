@pragma('vm:entry-point')
bool hasBalancedChargeRecoveries(String log) {
  final stack = <String>[];
  for (final ch in log.split('')) {
    if (ch.codeUnitAt(0) < 97) {
      stack.add(ch);
    } else if (stack.isEmpty || stack.removeLast() != ch.toUpperCase()) {
      return false;
    }
  }
  return stack.isEmpty;
}

@pragma('vm:entry-point')
void main() {
  assert(hasBalancedChargeRecoveries('') == true);
  assert(hasBalancedChargeRecoveries('ABba') == true);
  assert(hasBalancedChargeRecoveries('ABab') == false);
  print('All tests passed!');
}