import sys, urllib.request, urllib.parse, re
B="https://vesuvius-challenge-open-data.s3.amazonaws.com/"
def ls(prefix, delim="/"):
    out=[];tok=None
    while True:
        q={"list-type":"2","prefix":prefix,"delimiter":delim}
        if tok:q["continuation-token"]=tok
        x=urllib.request.urlopen(B+"?"+urllib.parse.urlencode(q)).read().decode()
        out+= [("D",p) for p in re.findall(r"<CommonPrefixes><Prefix>([^<]*)",x)]
        out+= [("F",k,int(s)) for k,s in re.findall(r"<Key>([^<]*)</Key>.*?<Size>(\d+)",x)]
        m=re.search(r"<NextContinuationToken>([^<]*)",x)
        if not m:break
        tok=m.group(1)
    return out
if __name__=="__main__":
    for e in ls(sys.argv[1], "" if len(sys.argv)>2 else "/"): print(*e)
