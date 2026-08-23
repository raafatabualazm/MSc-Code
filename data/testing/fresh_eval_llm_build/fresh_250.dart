@pragma('vm:entry-point')
int countEchoArtRows(String mural) {
  int total = 0;
  for (var row in mural.split('\n')) {
    var parts = row.trim().split(':');
    if (parts.length == 2 && parts[0].isNotEmpty && parts[0] == parts[1]) {
      total++;
    }
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(countEchoArtRows('@@:@@') == 1);
  assert(countEchoArtRows('@@:@@\n@@:@@@') == 1);
  assert(countEchoArtRows('') == 0);
  print('All tests passed!');
}