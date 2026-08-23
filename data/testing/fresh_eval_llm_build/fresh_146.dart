@pragma('vm:entry-point')
String rankBracketGroupSeedings(List<String> teamRecords, String groupName) {
  if (teamRecords.isEmpty) return '$groupName: (empty)';
  List<List<Object>> parsed = [];
  for (String record in teamRecords) {
    List<String> parts = record.split(':');
    String name = parts[0];
    int wins = int.parse(parts[1]);
    int pointDiff = int.parse(parts[3]);
    parsed.add([name, wins, pointDiff]);
  }
  parsed.sort((a, b) {
    int winsA = a[1] as int, winsB = b[1] as int;
    if (winsB != winsA) return winsB - winsA;
    int pdA = a[2] as int, pdB = b[2] as int;
    if (pdB != pdA) return pdB - pdA;
    return (a[0] as String).compareTo(b[0] as String);
  });
  StringBuffer sb = StringBuffer('$groupName: ');
  for (int i = 0; i < parsed.length; i++) {
    if (i > 0) sb.write(' ');
    sb.write('${i + 1}.${parsed[i][0]}');
  }
  return sb.toString();
}

@pragma('vm:entry-point')
void main() {
  assert(rankBracketGroupSeedings([], 'GroupA') == 'GroupA: (empty)');
  assert(rankBracketGroupSeedings(['Alpha:3:0:15', 'Beta:3:0:10', 'Gamma:2:1:5'], 'GroupA') == 'GroupA: 1.Alpha 2.Beta 3.Gamma');
  assert(rankBracketGroupSeedings(['X:1:2:-3', 'Y:2:1:4', 'Z:2:1:2'], 'Stage1') == 'Stage1: 1.Y 2.Z 3.X');
  print('All tests passed!');
}