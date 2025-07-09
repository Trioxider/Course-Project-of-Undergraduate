function zs=SBE(p)
z=[];
h=0;l=0;

m=length(p);
%Finding each alpha values 
a(1)=0;
for j=2:m
    a(j)=a(j-1)+p(j-1);
end

%Finding each code length
for i=1:m
    n(i)= ceil(-1*(log2(p(i))));
end

%Computing each code
for i=1:m
    int=a(i);
for j=1:n(i)
    frac=int*2;
    c=floor(frac);
    frac=frac-c;
    z=[z c];
    int=frac;
end
zs(i)=strrep(join(string(z(:))),' ','');
z=[];
end

