@pragma('vm:entry-point')
int locateQueueDrainPoint(List<String> logs) {
  int left = 0;
  int right = logs.length - 1;
  int start = logs.length;
  while (left <= right) {
    int mid = (left + right) >> 1;
    int cut = logs[mid].indexOf('|');
    int load = int.parse(cut == -1 ? logs[mid] : logs[mid].substring(0, cut));
    if (load >= 300) {
      start = mid;
      right = mid - 1;
    } else {
      left = mid + 1;
    }
  }
  for (int i = start; i < logs.length; i++) {
    int cut = logs[i].indexOf('|');
    int load = int.parse(cut == -1 ? logs[i] : logs[i].substring(0, cut));
    if (load > 450) {
      break;
    }
    if (logs[i].contains("drain")) {
      return i;
    }
  }
  return -1;
}

@pragma('vm:entry-point')
void main() {
  assert(locateQueueDrainPoint([]) == -1);
  assert(locateQueueDrainPoint(['120|boot','300|drain start']) == 1);
  assert(locateQueueDrainPoint(['100|ok','320|cache','451|drain']) == -1);
  print('All tests passed!');
}