@pragma('vm:entry-point')
bool isUrgentLogEntry(int logWord) {
  int severity = (logWord >> 28) & 0x7;
  int flags = logWord & 0x0FFFFFFF;
  int count = 0;
  int temp = flags;
  while (temp != 0) {
    count += temp & 1;
    temp >>= 1;
  }
  return severity == 7 && (count & 1) == 1;
}

@pragma('vm:entry-point')
void main() {
  assert(isUrgentLogEntry(0) == false);
  assert(isUrgentLogEntry(1879048193) == true);
  assert(isUrgentLogEntry(1879048195) == false);
  print('All tests passed!');
}