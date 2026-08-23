@pragma('vm:entry-point')
String morseDaySpanSummary(String morseSymbols, int startDay) {
  int dots = 0, dashes = 0, span = 0;
  for (int i = 0; i < morseSymbols.length; i++) {
    String ch = morseSymbols[i];
    if (ch == '.') {
      dots++;
      span += 1;
    } else if (ch == '-') {
      dashes++;
      span += 3;
    }
    if (i < morseSymbols.length - 1) {
      span += 1;
    }
  }
  int endDay = span == 0 ? startDay - 1 : startDay + span - 1;
  return 'span:$span,end:$endDay,dots:$dots,dashes:$dashes';
}

@pragma('vm:entry-point')
void main() {
  assert(morseDaySpanSummary('.-', 0) == 'span:5,end:4,dots:1,dashes:1');
  assert(morseDaySpanSummary('...', 5) == 'span:5,end:9,dots:3,dashes:0');
  assert(morseDaySpanSummary('', 0) == 'span:0,end:-1,dots:0,dashes:0');
  print('All tests passed!');
}