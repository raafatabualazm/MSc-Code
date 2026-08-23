@pragma('vm:entry-point')
int tideIntervalDriftScore(List<int> heights) {
  if (heights.isEmpty) return 0;
  int score = 0;
  for (int start = 0; start < heights.length; start++) {
    int level = heights[start];
    if (level < 0) {
      score -= start;
      continue;
    }
    int repeat = -1;
    for (int end = start + 1; end < heights.length; end++) {
      if (heights[end] == level) {
        repeat = end;
        break;
      }
      if ((end - start) > 4 && heights[end] < level) {
        score--;
      }
    }
    if (repeat == -1) {
      score += level % 5 == 0 ? start + 2 : -1;
      continue;
    }
    int gap = repeat - start;
    if (gap == 1) {
      score += level;
    } else if (gap % 2 == 0) {
      score += gap * 2;
    } else {
      score -= gap;
    }
    for (int mid = start + 1; mid < repeat; mid++) {
      if (heights[mid] > level) {
        score++;
      } else if (heights[mid] < 0) {
        score -= 2;
      }
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(tideIntervalDriftScore([]) == 0);
  assert(tideIntervalDriftScore([2, 3, 2]) == 3);
  assert(tideIntervalDriftScore([5, 1, 5, 1, 5]) == 18);
  print('All tests passed!');
}