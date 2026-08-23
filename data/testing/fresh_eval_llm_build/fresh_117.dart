@pragma('vm:entry-point')
bool isNickelRoundingNeutral(List<int> cents) {
  if (cents.isEmpty) return true;
  int balance = 0, driftGcd = 0;
  for (int i = 0; i < cents.length; i++) {
    int r = cents[i].abs() % 10;
    if (r == 0 || r == 5) continue;
    int drift;
    if (r < 3) {
      drift = -r;
    } else if (r < 5) {
      drift = 5 - r;
    } else {
      drift = r < 8 ? 5 - r : 10 - r;
    }
    balance += drift;
    int a = drift.abs();
    while (a != 0) {
      int t = driftGcd % a;
      driftGcd = a;
      a = t;
    }
    for (int j = 0; j < i; j++) {
      if (r == cents[j].abs() % 10 && (cents[i] < 0) != (cents[j] < 0)) return false;
    }
  }
  return balance == 0 && (driftGcd == 0 || driftGcd == 1);
}

@pragma('vm:entry-point')
void main() {
  assert(isNickelRoundingNeutral([101, 104]) == true);
  assert(isNickelRoundingNeutral([102, 108]) == false);
  assert(isNickelRoundingNeutral([-101, 111]) == false);
  print('All tests passed!');
}