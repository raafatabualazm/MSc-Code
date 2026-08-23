@pragma('vm:entry-point')
String sealManifestChecksum(List<int> weights) {
  StringBuffer sb = StringBuffer();
  for (int i = 0; i < weights.length - 1; i++) {
    int a = weights[i], b = weights[i+1];
    if (a < 0 || b < 0) return "INVALID";
    if (a == 0 && b == 0) continue;
    int g;
    if (a == 0) g = b.abs();
    else if (b == 0) g = a.abs();
    else {
      int x = a.abs(), y = b.abs();
      while (y != 0) { int t = y; y = x % y; x = t; }
      g = x;
    }
    if (g == 1) continue;
    bool prime = g > 1;
    for (int k = 2; k * k <= g; k++) { if (g % k == 0) { prime = false; break; } }
    if (prime) {
      String b36 = ''; int n = g;
      const d = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ';
      while (n > 0) { b36 = d[n % 36] + b36; n ~/= 36; }
      if (sb.isNotEmpty) sb.write('-');
      sb.write(b36);
    } else {
      if (sb.isNotEmpty) sb.write(':');
      sb.write(g);
    }
  }
  return sb.isEmpty ? "NONE" : sb.toString();
}

@pragma('vm:entry-point')
void main() {
  assert(sealManifestChecksum([2,4]) == "2");
  assert(sealManifestChecksum([0,0]) == "NONE");
  assert(sealManifestChecksum([-1,2]) == "INVALID");
  print('All tests passed!');
}