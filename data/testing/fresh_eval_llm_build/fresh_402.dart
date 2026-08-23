@pragma('vm:entry-point')
bool followsBatteryCycleTape(String tape) {
  int level = 0;
  for (final c in tape.split('')) {
    if (c == '+') level++;
    else if (c == '-') level--;
    else if (c != '|' || level != 0) return false;
    if (level < 0 || level > 2) return false;
  }
  return level == 0;
}

@pragma('vm:entry-point')
void main() {
  assert(followsBatteryCycleTape('+-|++--') == true);
  assert(followsBatteryCycleTape('+++---') == false);
  assert(followsBatteryCycleTape('+|-' ) == false);
  print('All tests passed!');
}