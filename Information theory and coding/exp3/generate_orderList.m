clear;
clc;

fid=fopen("English text data.txt","r");  %打开文件
count=zeros(1,27);
ff=fscanf(fid,'%c');
str=lower(ff(isletter(ff)|ff==' '));
fclose(fid);  %关闭行数据
orderList=charToOrder(str);  %将字母和空格转换为对应顺序编号