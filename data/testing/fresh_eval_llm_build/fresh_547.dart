@pragma('vm:entry-point')
String? mostFrequentSignalBin(String log) {
  if (log.isEmpty) return "No signals";
  Map<String,int> b={'Excellent':0,'Good':0,'Fair':0,'Poor':0,'No Signal':0};
  for(String t in log.split(';')){
    if(t.isEmpty) continue;
    int i=0; bool n=false;
    if(t[i]=='-'){n=true;i++;}
    int v=0; bool d=false;
    for(;i<t.length && t.codeUnitAt(i)>=48 && t.codeUnitAt(i)<=57;i++){
      d=true; v=v*10+(t.codeUnitAt(i)-48);
    }
    if(!d) return null;
    while(i<t.length && t[i]==' ') i++;
    if(i<t.length){
      if(i+3<=t.length && t.substring(i,i+3).toLowerCase()=='dbm') i+=3;
      else return null;
    }
    while(i<t.length && t[i]==' ') i++;
    if(i<t.length) return null;
    int s = n ? -v : v;
    if(s>= -50) b['Excellent']=b['Excellent']!+1;
    else if(s>=-60) b['Good']=b['Good']!+1;
    else if(s>=-70) b['Fair']=b['Fair']!+1;
    else if(s>=-80) b['Poor']=b['Poor']!+1;
    else b['No Signal']=b['No Signal']!+1;
  }
  String? r; int m=-1;
  for(String k in ['Excellent','Good','Fair','Poor','No Signal']){
    int c=b[k]!; if(c>m){m=c; r=k;}
  }
  return m==0?"No signals":r;
}

@pragma('vm:entry-point')
void main() {
  assert(mostFrequentSignalBin("-50") == "Excellent");
  assert(mostFrequentSignalBin("-55;-65") == "Good");
  assert(mostFrequentSignalBin("") == "No signals");
  print('All tests passed!');
}