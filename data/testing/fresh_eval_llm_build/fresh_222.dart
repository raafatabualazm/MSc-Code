@pragma('vm:entry-point')
bool hasAlternatingChannelDominance(String pixelData) {
  if (pixelData.isEmpty) return false;
  final pixelTokens = pixelData.split(';');
  if (pixelTokens.isEmpty) return false;
  for (int i = 0; i < pixelTokens.length; i++) {
    final token = pixelTokens[i].trim();
    if (token.isEmpty) return false;
    final parts = token.split(',');
    if (parts.length != 3) return false;
    int r = -1, g = -1, b = -1;
    for (int j = 0; j < 3; j++) {
      final val = int.tryParse(parts[j].trim());
      if (val == null || val < 0 || val > 255) return false;
      if (j == 0) r = val;
      else if (j == 1) g = val;
      else b = val;
    }
    final rDominant = r > g && r > b;
    final bDominant = b > g && b > r;
    if (i % 2 == 0) {
      if (!rDominant) return false;
    } else {
      if (!bDominant) return false;
    }
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(hasAlternatingChannelDominance("255,0,0") == true);
  assert(hasAlternatingChannelDominance("255,0,0;0,0,255") == true);
  assert(hasAlternatingChannelDominance("") == false);
  print('All tests passed!');
}