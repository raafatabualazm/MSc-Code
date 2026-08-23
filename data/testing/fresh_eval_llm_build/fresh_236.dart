@pragma('vm:entry-point')
String extractLightestContainerCode(String manifest) {
  double min = double.infinity;
  String code = "";
  for (var line in manifest.split('\n')) {
    String trimmed = line.trim();
    if (trimmed.isEmpty) continue;
    var parts = trimmed.split(RegExp(r'\s+'));
    double weight = double.parse(parts[1]);
    if (weight < min) {
      min = weight;
      code = parts[0];
    }
  }
  return code;
}

@pragma('vm:entry-point')
void main() {
  assert(extractLightestContainerCode("ABC 10\nDEF 5") == "DEF");
  assert(extractLightestContainerCode("") == "");
  assert(extractLightestContainerCode("XYZ 1.0\nABC 1.0") == "XYZ");
  print('All tests passed!');
}