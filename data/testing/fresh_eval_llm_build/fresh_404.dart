@pragma('vm:entry-point')
int countDominantPrecincts(List<List<int>> precinctVotes, int threshold) {
  int rows = precinctVotes.length;
  if (rows < 3) return 0;
  int cols = precinctVotes[0].length;
  if (cols < 3) return 0;
  int count = 0;
  for (int i = 1; i < rows - 1; i++) {
    for (int j = 1; j < cols - 1; j++) {
      int val = precinctVotes[i][j];
      if (val >= threshold &&
          val > precinctVotes[i-1][j] &&
          val > precinctVotes[i+1][j] &&
          val > precinctVotes[i][j-1] &&
          val > precinctVotes[i][j+1]) {
        count++;
      }
    }
  }
  return count;
}

@pragma('vm:entry-point')
void main() {
  assert(countDominantPrecincts([], 5) == 0);
  assert(countDominantPrecincts([[1,2,3],[4,5,6],[7,8,9]], 10) == 0);
  assert(countDominantPrecincts([[0,0,0],[0,1,0],[0,0,0]], 1) == 1);
  print('All tests passed!');
}