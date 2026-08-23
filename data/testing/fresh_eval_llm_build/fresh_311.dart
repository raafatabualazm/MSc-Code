@pragma('vm:entry-point')
String summarizeLogDigitsByBase(List<String> logLines, int base) {
  if (base < 2 || base > 16) throw ArgumentError('Base must be 2-16');
  int totalNonZeroDigits = 0;
  int linesProcessed = 0;
  for (var line in logLines) {
    var match = RegExp(r'\d+').firstMatch(line);
    if (match != null) {
      int num = int.parse(match.group(0)!);
      int nonZeroDigits = 0;
      while (num > 0) {
        if (num % base != 0) nonZeroDigits++;
        num ~/= base;
      }
      totalNonZeroDigits += nonZeroDigits;
      linesProcessed++;
    }
  }
  return 'Processed $linesProcessed log lines with base $base: total non-zero digits = $totalNonZeroDigits';
}

@pragma('vm:entry-point')
void main() {
  assert(summarizeLogDigitsByBase(["Error 503", "Info"], 10) == "Processed 1 log lines with base 10: total non-zero digits = 2");
  assert(summarizeLogDigitsByBase([], 2) == "Processed 0 log lines with base 2: total non-zero digits = 0");
  assert(summarizeLogDigitsByBase(["0"], 16) == "Processed 1 log lines with base 16: total non-zero digits = 0");
  print('All tests passed!');
}