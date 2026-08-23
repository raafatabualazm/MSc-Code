@pragma('vm:entry-point')
bool hasOutstandingSatellitePasses(List<String> passes) {
  Map<String, int> passCounts = {};
  Map<String, int> signalSums = {};
  const Set<String> blacklist = {'DEBRIS'};
  for (String pass in passes) {
    List<String> parts = pass.split(':');
    String name = parts[0];
    if (blacklist.contains(name)) continue;
    int signal = int.parse(parts[2]);
    passCounts[name] = (passCounts[name] ?? 0) + 1;
    signalSums[name] = (signalSums[name] ?? 0) + signal;
  }
  for (String name in passCounts.keys) {
    int count = passCounts[name]!;
    if (count >= 3 && signalSums[name]! / count > 12.5) {
      return true;
    }
  }
  return false;
}

@pragma('vm:entry-point')
void main() {
  assert(hasOutstandingSatellitePasses([]) == false);
  assert(hasOutstandingSatellitePasses(["A:1:13","A:1:13","A:1:13"]) == true);
  print('All tests passed!');
}