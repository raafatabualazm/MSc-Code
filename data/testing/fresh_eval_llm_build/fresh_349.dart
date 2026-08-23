@pragma('vm:entry-point')
String? decodeElevatorRunLength(String encoded, int maxFloor) {
  int starIndex = encoded.indexOf('*');
  if (starIndex == -1) return null;
  String dataPart = encoded.substring(0, starIndex);
  String checksumStr = encoded.substring(starIndex + 1);
  int? expectedChecksum = int.tryParse(checksumStr);
  if (expectedChecksum == null) return null;
  if (dataPart.isEmpty) {
    return expectedChecksum == 0 ? "" : null;
  }
  List<String> parts = dataPart.split(',');
  List<String> floors = [];
  int sum = 0;
  for (String part in parts) {
    List<String> pair = part.split(':');
    if (pair.length != 2) return null;
    int? floor = int.tryParse(pair[0]);
    int? count = int.tryParse(pair[1]);
    if (floor == null || count == null || floor <= 0 || count <= 0) return null;
    if (maxFloor > 0 && floor > maxFloor) return null;
    sum += floor * count;
    for (int i = 0; i < count; i++) {
      floors.add(floor.toString());
    }
  }
  if (sum % 10 != expectedChecksum) return null;
  return floors.join(',');
}

@pragma('vm:entry-point')
void main() {
  assert(decodeElevatorRunLength("2:3,5:1*1", 0) == "2,2,2,5");
  assert(decodeElevatorRunLength("*0", 0) == "");
  assert(decodeElevatorRunLength("2:1*3", 0) == null);
  print('All tests passed!');
}