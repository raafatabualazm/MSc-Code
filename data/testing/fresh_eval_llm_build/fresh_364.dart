@pragma('vm:entry-point')
List<num> encodeMazeSegments(String mazeRow) {
  if (mazeRow.isEmpty) return [];
  List<num> result = [];
  int count = 1;
  int prevCharCode = mazeRow.codeUnitAt(0);
  for (int i = 1; i < mazeRow.length; i++) {
    int currCharCode = mazeRow.codeUnitAt(i);
    if (currCharCode == prevCharCode) {
      count++;
    } else {
      switch (prevCharCode) {
        case 35:
          result.add(count);
          break;
        case 46:
          result.add(-count);
          break;
        case 83:
          result.add(count + 1000);
          break;
        case 69:
          result.add(count - 1000);
          break;
        default:
          throw ArgumentError('Invalid maze character: ${String.fromCharCode(prevCharCode)}');
      }
      count = 1;
      prevCharCode = currCharCode;
    }
  }
  switch (prevCharCode) {
    case 35:
      result.add(count);
      break;
    case 46:
      result.add(-count);
      break;
    case 83:
      result.add(count + 1000);
      break;
    case 69:
      result.add(count - 1000);
      break;
    default:
      throw ArgumentError('Invalid maze character: ${String.fromCharCode(prevCharCode)}');
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(encodeMazeSegments('').isEmpty);
  assert(encodeMazeSegments('#.#')[1] == -1);
  assert(encodeMazeSegments('S')[0] == 1001);
  print('All tests passed!');
}