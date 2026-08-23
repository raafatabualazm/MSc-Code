@pragma('vm:entry-point')
double qrModuleWindowSignal(List<int> modules) {
  double best = 0.0;
  int left = 0, dark = 0, light = 0, smudge = 0;
  for (int right = 0; right < modules.length; right++) {
    int v = modules[right];
    if (v == 1) {
      dark++;
    } else if (v == 0) {
      light++;
    } else {
      smudge++;
    }
    while (smudge > 1 || (dark - light).abs() > 2) {
      int drop = modules[left++];
      if (drop == 1) {
        dark--;
      } else if (drop == 0) {
        light--;
      } else {
        smudge--;
      }
    }
    int len = right - left + 1;
    if (len < 2) continue;
    int transitions = 0;
    for (int i = left + 1; i <= right; i++) {
      if (modules[i] == 2 || modules[i - 1] == 2) continue;
      if (modules[i] != modules[i - 1]) transitions++;
    }
    double score = len + transitions / 2.0;
    if (dark == light && smudge == 0) {
      score += 0.5;
    } else if (transitions == 0 && smudge == 1) {
      score -= 0.5;
    }
    if (score > best) best = score;
  }
  return best;
}

@pragma('vm:entry-point')
void main() {
  assert(qrModuleWindowSignal([]) == 0.0);
  assert(qrModuleWindowSignal([1, 0]) == 3.0);
  assert(qrModuleWindowSignal([2, 2]) == 0.0);
  print('All tests passed!');
}