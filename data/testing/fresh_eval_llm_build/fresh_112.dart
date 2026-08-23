@pragma('vm:entry-point')
bool hasBalancedRetryTokens(String line) {
  var parts = line.split('|');
  if (parts.length != 2 || parts[0].isEmpty) return false;
  var entries = parts[1].split(',');
  int balance = 0;
  for (var e in entries) {
    if (e.length < 2 || (e[0] != '+' && e[0] != '-')) return false;
    balance += e[0] == '+' ? 1 : -1;
  }
  return balance == 0 && entries[0].isNotEmpty;
}

@pragma('vm:entry-point')
void main() {
  assert(hasBalancedRetryTokens('edge-a|+cache,-db') == true);
  assert(hasBalancedRetryTokens('edge-b|+cache,+db') == false);
  assert(hasBalancedRetryTokens('|+cache,-db') == false);
  print('All tests passed!');
}