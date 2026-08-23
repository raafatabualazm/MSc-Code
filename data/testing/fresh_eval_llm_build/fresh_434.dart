@pragma('vm:entry-point')
List<String> scanTwoDigitBarcodeFrames(String stream) {
  List<String> out = [];
  String cur = '';
  bool open = false;
  for (int i = 0; i < stream.length; i++) {
    String c = stream[i];
    if (c == '!') { open = true; cur = ''; }
    else if (c == '?' && open) { if (cur.length == 2) out.add(cur); open = false; }
    else if (open && c.codeUnitAt(0) >= 48 && c.codeUnitAt(0) <= 57) { if (cur.length < 2) cur += c; else open = false; }
    else { open = false; }
  }
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(scanTwoDigitBarcodeFrames('!12?').toString() == '[12]');
  assert(scanTwoDigitBarcodeFrames('!1?!22?!333?').toString() == '[22]');
  assert(scanTwoDigitBarcodeFrames('').length == 0);
  print('All tests passed!');
}