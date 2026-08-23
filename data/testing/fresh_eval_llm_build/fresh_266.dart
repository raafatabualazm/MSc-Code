@pragma('vm:entry-point')
String gcdFromLogLineBase(String log, int base) {
  int gcd = 0;
  for (String token in log.split(RegExp(r'\s+|[,;]'))) {
    int? val = int.tryParse(token);
    if (val != null) {
      val = val.abs();
      if (gcd == 0) { gcd = val; }
      else {
        int a = gcd, b = val;
        while (b != 0) { int t = a % b; a = b; b = t; }
        gcd = a;
      }
    }
  }
  if (gcd == 0) return "0";
  if (base < 2 || base > 36) return "Invalid base";
  return gcd.toRadixString(base);
}

@pragma('vm:entry-point')
void main() {
  assert(gcdFromLogLineBase('1 2 3', 10) == '1');
  assert(gcdFromLogLineBase('abc', 10) == '0');
  assert(gcdFromLogLineBase('10 20', 99) == 'Invalid base');
  print('All tests passed!');
}