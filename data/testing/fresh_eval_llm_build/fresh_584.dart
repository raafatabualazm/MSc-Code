@pragma('vm:entry-point')
int countCleanCriticalClusters(List<String> logs) {
  if (logs.isEmpty) return 0;
  int h = logs.length, w = logs[0].length;
  var seen = List.generate(h, (_) => List.filled(w, false));
  int cnt = 0;
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      if (logs[i][j] != 'C' || seen[i][j]) continue;
      var q = <int>[];
      q.add(i * w + j);
      seen[i][j] = true;
      int sz = 1;
      bool eNear = false;
      while (q.isNotEmpty) {
        int p = q.removeAt(0);
        int r = p ~/ w, c = p % w;
        for (var d in [[-1,0],[1,0],[0,-1],[0,1]]) {
          int nr = r + d[0], nc = c + d[1];
          if (nr >= 0 && nr < h && nc >= 0 && nc < w) {
            if (logs[nr][nc] == 'E') {
              eNear = true;
            } else if (logs[nr][nc] == 'C' && !seen[nr][nc]) {
              seen[nr][nc] = true;
              q.add(nr * w + nc);
              sz++;
            }
          }
        }
      }
      if (sz >= 3 && !eNear) cnt++;
    }
  }
  return cnt;
}

@pragma('vm:entry-point')
void main() {
  assert(countCleanCriticalClusters([]) == 0);
  assert(countCleanCriticalClusters(["CCC"]) == 1);
  assert(countCleanCriticalClusters(["CC", "CE"]) == 0);
  print('All tests passed!');
}