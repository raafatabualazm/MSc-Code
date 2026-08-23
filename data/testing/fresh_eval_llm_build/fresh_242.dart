@pragma('vm:entry-point')
int compactLogErrorIndicators(List<int> logEntries) {
  int summary = 0, errorCount = 0, mod0Err = 0, sev7 = 0;
  bool high = false;
  for (int e in logEntries) {
    if ((e & 1) == 1) {
      errorCount++;
      if (((e >> 1) & 7) >= 4) high = true;
      if (((e >> 4) & 15) == 0) mod0Err++;
    }
    if (((e >> 1) & 7) == 7) sev7++;
  }
  summary |= high ? 1 : 0;
  summary |= (errorCount & 1) << 1;
  summary |= (mod0Err > 15 ? 15 : mod0Err) << 2;
  summary |= (sev7 > 15 ? 15 : sev7) << 6;
  return summary;
}

@pragma('vm:entry-point')
void main() {
  assert(compactLogErrorIndicators([]) == 0);
  assert(compactLogErrorIndicators([1]) == 6);
  assert(compactLogErrorIndicators([9]) == 7);
  print('All tests passed!');
}