@pragma('vm:entry-point')
int scoreRgbOrderingDrift(List<int> pixels, int tolerance) {
  if (pixels.isEmpty) return 0;
  List<int> sorted = List<int>.from(pixels);
  int score = 0;
  for (int i = 1; i < sorted.length; i++) {
    int key = sorted[i], kr = (key >> 16) & 255, kg = (key >> 8) & 255, kb = key & 255;
    int kBright = kr + kg + kb, j = i - 1;
    while (j >= 0) {
      int v = sorted[j], vr = (v >> 16) & 255, vg = (v >> 8) & 255, vb = v & 255;
      int vBright = vr + vg + vb;
      bool move = kBright + tolerance < vBright || ((kBright - vBright).abs() <= tolerance && (kg > vg || (kg == vg && kb < vb)));
      if (!move) break;
      sorted[j + 1] = v;
      score += (vr == kr || vb == kb) ? 2 : 1;
      j--;
    }
    sorted[j + 1] = key;
  }
  for (int i = 1; i < sorted.length; i++) {
    int a = sorted[i - 1], b = sorted[i], ar = (a >> 16) & 255, ag = (a >> 8) & 255, ab = a & 255;
    int br = (b >> 16) & 255, bg = (b >> 8) & 255, bb = b & 255;
    if (a == b) {
      score--;
      continue;
    }
    if ((ar - br).abs() <= tolerance && (ag - bg).abs() <= tolerance) score++;
    else if (((ar + ag + ab) - (br + bg + bb)).abs() > tolerance * 2) score += 3;
    if (ab > bb && ar < br) score += 2;
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(scoreRgbOrderingDrift([], 0) == 0);
  assert(scoreRgbOrderingDrift([0x010203, 0x010203], 0) == -1);
  assert(scoreRgbOrderingDrift([0x000001, 0x020000], 0) == 5);
  print('All tests passed!');
}