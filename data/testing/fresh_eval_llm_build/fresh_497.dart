@pragma('vm:entry-point')
String scaledRecipeSummary(List<int> amounts) {
  if (amounts.isEmpty) return "";
  int lcm = 1;
  for (int a in amounts) {
    int x = lcm, y = a;
    while (y != 0) {
      int t = y;
      y = x % y;
      x = t;
    }
    lcm = lcm * a ~/ x;
  }
  StringBuffer sb = StringBuffer();
  sb.write(lcm);
  sb.write(':');
  for (int i = 0; i < amounts.length; i++) {
    int scaled = lcm ~/ amounts[i];
    sb.write(scaled);
    int root = 1;
    while (root * root <= scaled) root++;
    root--;
    if (root * root == scaled) sb.write('*');
    if (i < amounts.length - 1) sb.write(',');
  }
  return sb.toString();
}

@pragma('vm:entry-point')
void main() {
  assert(scaledRecipeSummary([]) == "");
  assert(scaledRecipeSummary([6,8]) == "24:4*,3");
  assert(scaledRecipeSummary([1,2,3]) == "6:6,3,2");
  print('All tests passed!');
}