@pragma('vm:entry-point')
List<String> tokenizeMorseBursts(String tape) {
  List<String> out = [];
  String current = '';
  bool inNoise = false;
  int spaces = 0;
  for (int i = 0; i < tape.length; i++) {
    String c = tape[i];
    if (c == '.' || c == '-') {
      if (spaces > 1 && current.isNotEmpty) {
        out.add(current);
        out.add('<pause>');
        current = '';
      } else if (spaces == 1 && current.isNotEmpty) {
        out.add(current);
        current = '';
      }
      current += c;
      spaces = 0;
      inNoise = false;
    } else if (c == ' ') {
      spaces++;
    } else if (c == '/') {
      if (current.isNotEmpty) {
        out.add(current);
        current = '';
      }
      out.add('<gap>');
      spaces = 0;
      inNoise = false;
    } else {
      if (current.isNotEmpty) {
        out.add(current);
        current = '';
      }
      if (!inNoise) out.add('<noise>');
      inNoise = true;
      spaces = 0;
    }
  }
  if (current.isNotEmpty) out.add(current);
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(tokenizeMorseBursts('').toString() == '[]');
  assert(tokenizeMorseBursts('.  -').toString() == '[., <pause>, -]');
  assert(tokenizeMorseBursts('..x--').toString() == '[.., <noise>, --]');
  print('All tests passed!');
}