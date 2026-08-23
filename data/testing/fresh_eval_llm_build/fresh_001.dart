@pragma('vm:entry-point')
String scaleRecipePinches(String pinches, int batches) {
  if (pinches.isEmpty) {
    return '';
  }
  int amount = (pinches.codeUnitAt(0) - 48) * batches;
  String rest = scaleRecipePinches(pinches.substring(1), batches);
  return rest.isEmpty ? amount.toString() : '${amount.toString()}:${rest}';
}

@pragma('vm:entry-point')
void main() {
  assert(scaleRecipePinches('', 3) == '');
  assert(scaleRecipePinches('5', 2) == '10');
  assert(scaleRecipePinches('203', -1) == '-2:0:-3');
  print('All tests passed!');
}