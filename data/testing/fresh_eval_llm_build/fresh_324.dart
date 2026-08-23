@pragma('vm:entry-point')
double decodeShiftedMoistureAverage(String encodedGrid, int shiftKey) {
  int i=0, sum=0, total=0;
  while(i<encodedGrid.length){
    var c=encodedGrid[i];
    if(c=='M'){
      int m=(encodedGrid.codeUnitAt(i+1)-48)*10, cnt=encodedGrid.codeUnitAt(i+2)-65+1;
      int s=(m+shiftKey)%100; if(s<0) s+=100;
      sum+=s*cnt; total+=cnt; i+=3;
    }else if(c=='D'){
      int cnt=encodedGrid.codeUnitAt(i+1)-65+1;
      int s=(shiftKey)%100; if(s<0) s+=100;
      sum+=s*cnt; total+=cnt; i+=2;
    }else{i++;}
  }
  return total==0?0.0:sum/total;
}

@pragma('vm:entry-point')
void main() {
  assert(decodeShiftedMoistureAverage('M5A', 0) == 50.0);
  assert(decodeShiftedMoistureAverage('DA', 0) == 0.0);
  assert(decodeShiftedMoistureAverage('', 99) == 0.0);
  print('All tests passed!');
}