@pragma('vm:entry-point')
String auditNickelRoundingPattern(List<int> cents) {
  int g = 0;
  int checksum = 0;
  const digits = '0123456789abcdefghijklmnopqrstuvwxyz';
  for (int v in cents) {
    int rem = v.abs() % 5;
    if (rem == 0) {
      checksum = (checksum + 1) % 36;
      continue;
    }
    int adjust = rem <= 2 ? rem : 5 - rem;
    checksum = (checksum * 7 + (v < 0 ? -adjust : adjust)) % 36;
    if (checksum < 0) checksum += 36;
    int a = g, b = adjust;
    while (b != 0) {
      int t = a % b;
      a = b;
      b = t;
    }
    g = a;
  }
  return '$g:${digits[checksum]}';
}

@pragma('vm:entry-point')
void main() {
  assert(auditNickelRoundingPattern([]) == '0:0');
  assert(auditNickelRoundingPattern([2, 3]) == '2:g');
  assert(auditNickelRoundingPattern([-6, 7]) == '1:v');
  print('All tests passed!');
}