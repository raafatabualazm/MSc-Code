@pragma('vm:entry-point')
double balancedMoistureDrift(List<List<int>> grid) {
  double score = 0.0;
  for (final row in grid) {
    int wet = 0;
    int dry = 0;
    for (final cell in row) {
      if (cell >= 8) {
        wet++;
      } else if (cell <= 2) {
        dry++;
      }
      if (cell.isOdd) {
        score += 0.5;
      } else {
        score -= 0.25;
      }
    }
    if (wet > dry) {
      score += (wet - dry).toDouble();
    } else if (dry > wet) {
      score -= (dry - wet) * 0.5;
    } else if (row.isNotEmpty) {
      score += 0.25;
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(balancedMoistureDrift([]) == 0.0);
  assert(balancedMoistureDrift([[8]]) == 0.75);
  assert(balancedMoistureDrift([[4, 6]]) == -0.25);
  print('All tests passed!');
}