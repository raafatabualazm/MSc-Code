@pragma('vm:entry-point')
int countQuietMazePockets(List<String> maze, int maxSize) {
  if (maze.isEmpty) return 0;
  int h = maze.length, w = maze[0].length, score = 0;
  List<List<bool>> seen = List.generate(h, (_) => List.filled(w, false));
  int dfs(int r, int c, List<bool> edge) {
    if (r < 0 || c < 0 || r >= h || c >= w || maze[r][c] == '#' || seen[r][c]) return 0;
    seen[r][c] = true;
    if (r == 0 || c == 0 || r == h - 1 || c == w - 1) edge[0] = true;
    return 1 + dfs(r + 1, c, edge) + dfs(r - 1, c, edge) + dfs(r, c + 1, edge) + dfs(r, c - 1, edge);
  }
  for (int r = 0; r < h; r++) {
    for (int c = 0; c < w; c++) {
      if (maze[r][c] == '.' && !seen[r][c]) {
        List<bool> edge = [false];
        int size = dfs(r, c, edge);
        if (!edge[0] && size <= maxSize) score += size;
      }
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(countQuietMazePockets([], 2) == 0);
  assert(countQuietMazePockets(['###','#.#','###'], 1) == 1);
  assert(countQuietMazePockets(['#####','#...#','#####'], 2) == 0);
  print('All tests passed!');
}