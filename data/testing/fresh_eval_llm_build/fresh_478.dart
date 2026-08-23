@pragma('vm:entry-point')
bool acceptsDnaRelayScript(String dna, String script) {
  int i = 0;
  int j = 0;
  bool requireFlip = false;
  while (j < script.length) {
    String token = script[j];
    if (token == '^') {
      requireFlip = true;
      j++;
      continue;
    }
    if (i >= dna.length) {
      return false;
    }
    String base = dna[i];
    if ('ACGT'.indexOf(base) < 0) {
      return false;
    }
    if (requireFlip && i > 0 && base == dna[i - 1]) {
      return false;
    }
    requireFlip = false;
    if (token == 'N') {
      i++;
      j++;
      continue;
    }
    if ('ACGT'.indexOf(token) < 0 || base != token) {
      return false;
    }
    i++;
    j++;
    if (j < script.length && script[j] == '+') {
      int extra = 0;
      while (i < dna.length && dna[i] == token) {
        extra++;
        i++;
      }
      if (extra == 0) {
        return false;
      }
      j++;
    }
  }
  return i == dna.length && !requireFlip;
}

@pragma('vm:entry-point')
void main() {
  assert(acceptsDnaRelayScript('AACT', 'A+CT') == true);
  assert(acceptsDnaRelayScript('ACT', 'A+CT') == false);
  assert(acceptsDnaRelayScript('AG', 'A^N') == true);
  print('All tests passed!');
}