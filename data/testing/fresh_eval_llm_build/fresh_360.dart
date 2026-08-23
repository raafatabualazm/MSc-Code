@pragma('vm:entry-point')
bool clearsQuayFiveManifest(String manifest) {
  int remainder = 0;
  int digits = 0;
  for (int i = 0; i < manifest.length; i++) {
    int code = manifest.codeUnitAt(i);
    if (code == 45) continue;
    int value = code >= 65 ? code - 55 : code - 48;
    remainder = (remainder * 14 + value) % 5;
    digits++;
  }
  return digits > 0 && remainder == 0;
}

@pragma('vm:entry-point')
void main() {
  assert(clearsQuayFiveManifest('5') == true);
  assert(clearsQuayFiveManifest('10') == false);
  assert(clearsQuayFiveManifest('1A9') == true);
  print('All tests passed!');
}