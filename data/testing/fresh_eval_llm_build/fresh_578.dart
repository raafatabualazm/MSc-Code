@pragma('vm:entry-point')
bool barcodeDigitIntervalParity(String ledger) {
  if (ledger.isEmpty) return true;
  List<String> parts = ledger.split('|');
  List<int> finals = [];
  List<int> starts = [];
  for (int i = 0; i < parts.length; i++) {
    String part = parts[i];
    if (part.isEmpty) return false;
    int day = 0;
    List<int> seen = [0];
    for (int j = 0; j < part.length; j++) {
      int v = part.codeUnitAt(j) - 48;
      if (v < 0 || v > 9) return false;
      if (v == 0 && j > 0) return false;
      day += v;
      for (int k = 0; k < seen.length; k++) {
        if (seen[k] == day) return false;
      }
      seen.add(day);
    }
    if ((i.isEven && day.isOdd) || (i.isOdd && day.isEven)) return false;
    finals.add(day);
    starts.add(part.codeUnitAt(0));
  }
  for (int i = 0; i < finals.length; i++) {
    for (int j = i + 1; j < finals.length; j++) {
      if (finals[i] == finals[j] && starts[i] == starts[j]) return false;
    }
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(barcodeDigitIntervalParity('') == true);
  assert(barcodeDigitIntervalParity('22|111') == true);
  assert(barcodeDigitIntervalParity('22|111|22') == false);
  print('All tests passed!');
}