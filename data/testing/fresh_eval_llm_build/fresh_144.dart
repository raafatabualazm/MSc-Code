@pragma('vm:entry-point')
bool validateTelemetryRlePacket(String encoded) {
  int hashIdx = encoded.indexOf('#');
  if (hashIdx < 0) return false;
  String rle = encoded.substring(0, hashIdx);
  String checksumStr = encoded.substring(hashIdx + 1);
  if (checksumStr.isEmpty) return false;
  int? checksum = int.tryParse(checksumStr);
  if (checksum == null) return false;
  Map<String, int> counts = {};
  int total = 0;
  int i = 0;
  while (i < rle.length) {
    int digitStart = i;
    while (i < rle.length && rle[i].compareTo('0') >= 0 && rle[i].compareTo('9') <= 0) {
      i++;
    }
    if (i == digitStart || i >= rle.length) return rle.isEmpty && checksum == 0;
    int? count = int.tryParse(rle.substring(digitStart, i));
    if (count == null || count < 1) return false;
    String sensor = rle[i];
    if (sensor.compareTo('A') < 0 || sensor.compareTo('Z') > 0) return false;
    counts[sensor] = (counts[sensor] ?? 0) + count;
    total += count;
    i++;
  }
  if (total != checksum) return false;
  if (total == 0) return true;
  for (String key in counts.keys) {
    if (counts[key]! * 10 > total * 4) return false;
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(validateTelemetryRlePacket('3A2B3C#8') == true);
  assert(validateTelemetryRlePacket('3A2B5C#10') == false);
  assert(validateTelemetryRlePacket('') == false);
  print('All tests passed!');
}