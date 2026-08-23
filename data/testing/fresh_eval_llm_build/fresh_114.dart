@pragma('vm:entry-point')
List<int> precinctWeightedMargins(List<List<int>> precincts) {
  if (precincts.isEmpty) return [];
  int totalA = 0, totalB = 0, sumX = 0, sumY = 0;
  int count = precincts.length;
  for (var p in precincts) {
    totalA += p[2];
    totalB += p[3];
    sumX += p[0];
    sumY += p[1];
  }
  int centX = sumX ~/ count;
  int centY = sumY ~/ count;
  List<int> result = [];
  for (var p in precincts) {
    int margin = p[2] - p[3];
    int dist = (p[0] - centX).abs() + (p[1] - centY).abs();
    int weight;
    if (dist <= 5) {
      weight = 3;
    } else if (dist <= 10) {
      weight = 2;
    } else if (dist <= 20) {
      weight = 1;
    } else {
      weight = 0;
    }
    result.add(margin * weight);
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(precinctWeightedMargins([]).toString() == '[]');
  assert(precinctWeightedMargins([[0,0,10,5]])[0] == 15);
  assert(precinctWeightedMargins([[0,0,10,5],[5,5,15,10],[10,10,20,15]]).toString() == '[10, 15, 10]');
  print('All tests passed!');
}