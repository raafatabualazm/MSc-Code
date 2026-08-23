@pragma('vm:entry-point')
List<String> classifyRoundedCentSheets(List<List<int>> ledger, int unit) {
  if (unit <= 0) {
    return [];
  }
  List<String> result = [];
  int threshold = unit ~/ 2;
  for (List<int> row in ledger) {
    if (row.isEmpty) {
      result.add('skip');
      continue;
    }
    int roundedTotal = 0;
    int adjusted = 0;
    int rejected = 0;
    for (int cents in row) {
      if (cents.abs() < threshold) {
        rejected++;
        continue;
      }
      int rem = cents.abs() % unit;
      int rounded = cents.abs() - rem;
      if (rem * 2 >= unit) {
        rounded += unit;
      }
      rounded = cents < 0 ? -rounded : rounded;
      if (rounded != cents) {
        adjusted++;
      }
      if (rounded == 0 && cents != 0) {
        rejected++;
      }
      roundedTotal += rounded;
    }
    if (rejected == row.length) {
      result.add('void');
    } else if (roundedTotal < 0) {
      result.add('debt:$roundedTotal:$adjusted');
    } else if (roundedTotal == 0) {
      result.add('flat:$adjusted');
    } else {
      result.add('credit:$roundedTotal:$adjusted');
    }
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(classifyRoundedCentSheets([[2, -2]], 4).first == 'flat:2');
  assert(classifyRoundedCentSheets([[]], 5).toString() == '[skip]');
  assert(classifyRoundedCentSheets([[9, 10, 11]], 10).length == 1);
  print('All tests passed!');
}