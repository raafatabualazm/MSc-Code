@pragma('vm:entry-point')
String hashBucketCycleSummary(List<int> dailyDelta) {
  int balance = 0;
  int longest = 0;
  int current = 0;
  int pressureDays = 0;
  int highRun = 0;
  for (int i = 0; i < dailyDelta.length; i++) {
    balance += dailyDelta[i];
    current++;
    if (balance < 0) {
      return 'invalid@$i';
    } else if (balance >= 10) {
      highRun++;
      if (highRun >= 2) {
        pressureDays++;
      }
    } else {
      highRun = 0;
      if (balance == 0) {
        if (current > longest) {
          longest = current;
        }
        current = 0;
      }
    }
  }
  return '${balance == 0 ? 'closed' : 'open'}|$longest|$pressureDays|$balance';
}

@pragma('vm:entry-point')
void main() {
  assert(hashBucketCycleSummary([]) == 'closed|0|0|0');
  assert(hashBucketCycleSummary([1, -1]) == 'closed|2|0|0');
  assert(hashBucketCycleSummary([-1]) == 'invalid@0');
  print('All tests passed!');
}