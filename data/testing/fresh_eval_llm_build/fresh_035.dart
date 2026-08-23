@pragma('vm:entry-point')
String wifiIndexShiftEncode(String signal) {
  if (signal.isEmpty) return "";
  var result = StringBuffer();
  for (int i = 0; i < signal.length; i++) {
    int digit = int.parse(signal[i]);
    result.write(((digit + i) % 10).toString());
  }
  return result.toString();
}

@pragma('vm:entry-point')
void main() {
  assert(wifiIndexShiftEncode("") == "");
  assert(wifiIndexShiftEncode("12") == "13");
  assert(wifiIndexShiftEncode("999") == "901");
  print('All tests passed!');
}