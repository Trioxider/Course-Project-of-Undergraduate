function dict=fano(q)
q=q';
minq=min(q);
for i=1:1:length(q)
    for j=1:1:length(q)
        if(q(j)==q(i))
            q(j)=q(j)+minq/10000*j;
        end
    end
end

p=flipud(sort(q));
bf=zeros(length(p),1);

length(p);
for i=1:1:length(p)
    i;    
    cond=0;
    count=1;
        p=flipud(sort(q));
        pq=p;
        while(cond==0)
            [res1,res2,size1,size2]=splitf(pq);
            clear find1
            clear find2
            find1=find(res1==p(i));
            find2=find(res2==p(i));
            sf1=size(find1);
            sf2=size(find2);
            if(sf1(1,1)>0)
                if  (count==1)
                    b(i)=2;
                else
                    b(i)=b(i)*10+2;
                end
                if(size1==1)
                    cond=1;
                    bf(i)=b(i);
                else
                    clear pq
                    pq=res1;
                end
            elseif(sf2(1,1)>0)
                if  (count==1)
                    b(i)=1;
                else
                    b(i)=b(i)*10+1;
                end
                
                if(size2==1)
                    cond=1;
                    bf(i)=b(i);
                else
                    clear pq
                    pq=res2;
                end
            end
        count=count+1;    
        end
end

for i=1:1:length(bf)
    if (i==1)
        s=two2o(bf(i));
    else
        s=strcat(s,two2o(bf(i)));
    end
end
s(end)=' ';
dict=strsplit(s,"-");