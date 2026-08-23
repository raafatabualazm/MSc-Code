@pragma('vm:entry-point')
int parseInventoryTapeValue(String tape, int capacity) {
  int used = 0;
  int score = 0;
  int i = 0;
  bool locked = false;
  while (i < tape.length) {
    String c = tape[i];
    if (c == ' ' || c == ',') {
      i++;
      continue;
    }
    if (c == '!') {
      locked = !locked;
      i++;
      continue;
    }
    int sign = c == '+' ? 1 : (c == '-' ? -1 : 0);
    if (sign == 0) return -1;
    i++;
    int rarity = 0;
    while (i < tape.length) {
      int code = tape.codeUnitAt(i);
      if (code >= 65 && code <= 90) {
        rarity += (code - 64) % 5;
        i++;
      } else {
        break;
      }
    }
    if (rarity == 0 || i >= tape.length || tape[i] != ':') return -1;
    i++;
    int qty = 0;
    int digits = 0;
    while (i < tape.length) {
      int d = tape.codeUnitAt(i) - 48;
      if (d < 0 || d > 9) break;
      qty = qty * 10 + d;
      digits++;
      i++;
    }
    if (digits == 0) return -1;
    int delta = qty + rarity;
    if (locked && sign > 0) {
      score -= delta;
      continue;
    }
    if (sign > 0) {
      if (used + delta > capacity) return score;
      used += delta;
      score += qty;
    } else if (delta > used) {
      used = 0;
      score--;
    } else {
      used -= delta;
      score += rarity;
    }
  }
  return score + used;
}

@pragma('vm:entry-point')
void main() {
  assert(parseInventoryTapeValue('', 5) == 0);
  assert(parseInventoryTapeValue('+A:2,-B:1', 10) == 4);
  assert(parseInventoryTapeValue('!+A:2', 10) == -3);
  print('All tests passed!');
}