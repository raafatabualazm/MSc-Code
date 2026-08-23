@pragma('vm:entry-point')
int evaluateLootSatchels(String log) {
  int score = 0, items = 0;
  bool open = false, cursed = false;
  for (final c in log.split('')) {
    if (c == '{') { open = true; cursed = false; items = 0; }
    else if (open && c == '}') { score += cursed ? -items : (items == 2 ? 2 : 0); open = false; }
    else if (open && c == '#') cursed = true;
    else if (open && c.codeUnitAt(0) >= 97 && c.codeUnitAt(0) <= 122) items++;
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(evaluateLootSatchels("{ab}") == 2);
  assert(evaluateLootSatchels("{ab#}") == -2);
  assert(evaluateLootSatchels("{a}{bc}") == 2);
  print('All tests passed!');
}