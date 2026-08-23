@pragma('vm:entry-point')
String firstLedgerDateExceedingBalance(List<int> dailyTransactions, int startingBalance, int threshold) {
  if (startingBalance > threshold) return 'day 0';
  int lo = 0;
  int hi = dailyTransactions.length - 1;
  // First build prefix sums to enable binary search
  List<int> prefix = [];
  int running = startingBalance;
  for (int i = 0; i < dailyTransactions.length; i++) {
    running += dailyTransactions[i];
    prefix.add(running);
  }
  // Binary search for leftmost index where prefix[i] > threshold
  int result = -1;
  while (lo <= hi) {
    int mid = (lo + hi) ~/ 2;
    if (prefix[mid] > threshold) {
      result = mid;
      hi = mid - 1;
    } else {
      lo = mid + 1;
    }
  }
  if (result == -1) return 'never';
  return 'day ${result + 1}';
}

@pragma('vm:entry-point')
void main() {
  assert(firstLedgerDateExceedingBalance([10, 20, 30], 0, 25) == 'day 2');
  assert(firstLedgerDateExceedingBalance([5, 5, 5], 0, 20) == 'never');
  assert(firstLedgerDateExceedingBalance([], 100, 50) == 'day 0');
  print('All tests passed!');
}