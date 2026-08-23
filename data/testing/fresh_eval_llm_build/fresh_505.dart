@pragma('vm:entry-point')
int scaleRecipeFromRLE(String rle) {
  if (rle.isEmpty) return 0;
  int scale = 1;
  int i = 0;
  if (rle[0] == 's') {
    if (rle.length == 1) return 0;
    scale = int.parse(rle[1]);
    i = 2;
  }
  int total = 0;
  while (i < rle.length) {
    if (i + 1 >= rle.length) break;
    total += scale * int.parse(rle[i + 1]);
    i += 2;
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(scaleRecipeFromRLE('a3') == 3);
  assert(scaleRecipeFromRLE('s2a3') == 6);
  assert(scaleRecipeFromRLE('') == 0);
  print('All tests passed!');
}