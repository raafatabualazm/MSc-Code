@pragma('vm:entry-point')
int countStableScaledIngredients(int recipePlan) {
  int doubles = recipePlan & 0xFF;
  int fragile = (recipePlan >> 8) & 0xFF;
  int rotated = ((doubles << 1) | (doubles >> 7)) & 0xFF;
  int combined = (rotated & ~fragile) | (fragile << 8);
  int total = 0;
  while (combined != 0) {
    total += combined & 1;
    combined >>= 1;
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(countStableScaledIngredients(0) == 0);
  assert(countStableScaledIngredients(257) == 2);
  assert(countStableScaledIngredients(21930) == 4);
  print('All tests passed!');
}