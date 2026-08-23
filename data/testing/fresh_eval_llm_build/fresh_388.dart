@pragma('vm:entry-point')
List<String> classifyManifestChecks(List<int> manifests) {
  List<String> out = [];
  for (int value in manifests) {
    if (value == 0) {
      out.add('hold');
      continue;
    }
    int n = value.abs();
    String base = '';
    int temp = n;
    while (temp > 0) {
      base = '${temp % 7}$base';
      temp ~/= 7;
    }
    int gcdDigit = 0;
    for (int i = 0; i < base.length; i++) {
      int d = base.codeUnitAt(i) - 48;
      if (d == 0) continue;
      if (gcdDigit == 0) {
        gcdDigit = d;
      } else {
        int a = gcdDigit, b = d;
        while (b != 0) {
          int t = a % b;
          a = b;
          b = t;
        }
        gcdDigit = a;
      }
    }
    bool prime = n > 1;
    for (int d = 2; d * d <= n; d++) {
      if (n % d == 0) {
        prime = false;
        break;
      }
    }
    if (prime && gcdDigit > 1) {
      out.add('seal:$base');
    } else if (value < 0 || gcdDigit == 1) {
      out.add('split:$base');
    } else {
      out.add('hold:$base');
    }
  }
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(classifyManifestChecks([2, 14]).toString() == '[seal:2, hold:20]');
  assert(classifyManifestChecks([0, -14]).length == 2);
  assert(classifyManifestChecks([49]).toString() == '[split:100]');
  print('All tests passed!');
}