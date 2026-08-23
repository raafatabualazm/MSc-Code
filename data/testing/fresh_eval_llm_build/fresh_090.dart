@pragma('vm:entry-point')
String sortMazeCellsByProximity(List<String> cellEntries, String centerCell) {
  var cp = centerCell.split(','); int cx = int.parse(cp[0]), cy = int.parse(cp[1]);
  var cats = <int>[], dists = <int>[];
  for (var e in cellEntries) {
    var p = e.split(','); int x = int.parse(p[0]), y = int.parse(p[1]);
    int dx = (x-cx).abs(), dy = (y-cy).abs();
    cats.add(x==cx&&y==cy?0: y==cy?1: x==cx?2: dx==dy?3:4);
    dists.add(dx+dy);
  }
  var indices = List<int>.generate(cellEntries.length, (i)=>i);
  indices.sort((i,j){
    if(cats[i]!=cats[j]) return cats[i]-cats[j];
    if(dists[i]!=dists[j]) return dists[i]-dists[j];
    return cellEntries[i].compareTo(cellEntries[j]);
  });
  return indices.map((i)=>cellEntries[i]).join(',');
}

@pragma('vm:entry-point')
void main() {
  assert(sortMazeCellsByProximity([], "0,0") == "");
  assert(sortMazeCellsByProximity(["0,0"], "0,0") == "0,0");
  assert(sortMazeCellsByProximity(["1,0", "0,1"], "0,0") == "1,0,0,1");
  print('All tests passed!');
}