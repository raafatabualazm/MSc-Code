@pragma('vm:entry-point')
List<int> collectMazePocketAreas(List<String> maze, int minWalls) {
  if (maze.isEmpty || maze[0].isEmpty) return [];
  int h = maze.length, w = maze[0].length;
  List<List<bool>> seen = List.generate(h, (_) => List.filled(w, false));
  List<int> areas = [];
  List<int> dfs(int r, int c) {
    seen[r][c] = true;
    int area = 1, walls = 0, border = 0;
    for (List<int> d in const [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      int nr = r + d[0], nc = c + d[1];
      if (nr < 0 || nc < 0 || nr >= h || nc >= w) {
        border = 1;
        continue;
      }
      if (maze[nr][nc] == '#') {
        walls++;
        continue;
      }
      if (seen[nr][nc]) continue;
      List<int> sub = dfs(nr, nc);
      area += sub[0];
      walls += sub[1];
      if (sub[2] == 1) border = 1;
    }
    return [area, walls, border];
  }

  for (int r = 0; r < h; r++) {
    for (int c = 0; c < w; c++) {
      if (maze[r][c] == '#' || seen[r][c]) continue;
      List<int> info = dfs(r, c);
      if (info[2] == 1) continue;
      if (info[1] >= minWalls) areas.add(info[0]);
    }
  }
  areas.sort();
  return areas;
}

@pragma('vm:entry-point')
void main() {
  assert(collectMazePocketAreas([], 1).toString() == '[]');
  assert(collectMazePocketAreas(['###','#.#','###'], 4).toString() == '[1]');
  assert(collectMazePocketAreas(['#######','#..#..#','#######'], 6).length == 2);
  print('All tests passed!');
}