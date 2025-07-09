clear all
clc
music=audiorecorder(8000,16,1);
%创建一个保存音频信息的对象，它包含采样率，时间和录制的音频信息等等。
%8000表示采样为8000Hz（可改为8000, 11025, 22050, 44100等，
%此数值越大，录入的声音质量越好，相应需要的存储空间越大）
%16为用16bits存储，1为单通道即单声道（也可以改为2双通道即单声道）。

recordblocking(music,5);
%开始录制，此时对着麦克风说话即可,录制时间为5秒。
play(music);

MyRecording=getaudiodata(music);
%得到以n*2列数字矩阵存储的刚录制的音频信号。
plot(MyRecording);

filename='audio.wav';
audiowrite(filename,MyRecording,8000);
%MyRecording表示要存入的波形矩阵，
%8000表采样率，'audio'为存储的文件名
