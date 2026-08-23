@pragma('vm:entry-point')
bool hasCoprimeLogCodes(String logs) {
  int primeCount = 0, compositeCount = 0, current = -1;
  for (int i = 0; i <= logs.length; i++) {
    int ch = i < logs.length ? logs.codeUnitAt(i) : 35;
    if (i < logs.length && ch >= 48 && ch <= 57) {
      current = current < 0 ? ch - 48 : current * 10 + ch - 48;
    } else if (current >= 0) {
      if (current > 1) {
        bool prime = true;
        for (int d = 2; d * d <= current; d++) {
          if (current % d == 0) { prime = false; break; }
        }
        if (prime) { primeCount++; } else { compositeCount++; }
      }
      current = -1;
    }
  }
  if (primeCount + compositeCount == 0) return false;
  int a = primeCount, b = compositeCount;
  while (b != 0) { int t = a % b; a = b; b = t; }
  return a == 1;
}

@pragma('vm:entry-point')
void main() {
  assert(hasCoprimeLogCodes("") == false);
  assert(hasCoprimeLogCodes("api7warn4") == true);
  assert(hasCoprimeLogCodes("id2x3y4z6") == false);
  print('All tests passed!');
}