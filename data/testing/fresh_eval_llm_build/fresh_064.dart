@pragma('vm:entry-point')
int countSeedUpsets(String bracketResult, int minSeedThreshold) {
  if (bracketResult.isEmpty) return 0;
  int upsets = 0;
  int i = 0;
  // skip leading '['
  while (i < bracketResult.length && bracketResult[i] == '[') i++;
  while (i < bracketResult.length) {
    if (bracketResult[i] == ']') break;
    // parse first seed number
    int numStart = i;
    while (i < bracketResult.length && bracketResult[i].codeUnitAt(0) >= 48 && bracketResult[i].codeUnitAt(0) <= 57) i++;
    if (i == numStart) { i++; continue; }
    int seedA = int.parse(bracketResult.substring(numStart, i));
    // expect 'v'
    if (i >= bracketResult.length || bracketResult[i] != 'v') { i++; continue; }
    i++; // skip 'v'
    int numStart2 = i;
    while (i < bracketResult.length && bracketResult[i].codeUnitAt(0) >= 48 && bracketResult[i].codeUnitAt(0) <= 57) i++;
    if (i == numStart2) { i++; continue; }
    int seedB = int.parse(bracketResult.substring(numStart2, i));
    // winner is seedA (first listed); upset if seedA > seedB and seedA >= minSeedThreshold
    if (seedA > seedB && seedA >= minSeedThreshold) upsets++;
    // skip comma or other separator
    while (i < bracketResult.length && bracketResult[i] == ',') i++;
  }
  return upsets;
}

@pragma('vm:entry-point')
void main() {
  assert(countSeedUpsets("", 1) == 0);
  assert(countSeedUpsets("[3v1,2v4]", 1) == 1);
  assert(countSeedUpsets("[8v1,7v2,6v3]", 7) == 2);
  print('All tests passed!');
}