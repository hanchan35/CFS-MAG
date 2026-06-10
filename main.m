clear;
clc;
close all;
addpath(genpath(pwd));
%读入数据
%Features:Samples

% %wine,13:178,con
% load("wine.mat");
% dataName="wine";

% %congress,16:435,dis
% load("congress.mat");
% dataName="congress";

% %soybean-small,35:47,dis
% load("soybean-small.mat");
% dataName="soybean-small";

% %Splice,60:3190,dis
% load("Splice.mat");
% dataName="Splice";

% %musk1,166:476,con
% load("musk1.mat");
% dataName="musk1";

% %musk2,166:6598,con
% load("musk2.mat");
% dataName="musk2";

% %darwin,450:174,con
% load("darwin.mat");
% dataName="darwin";

% %COIL20,1024:1440,con
% load("COIL20.mat");
% dataName="COIL20";

% %colon,2000:62,dis
% load("colon.mat");
% dataName="colon";

% %PCMAC,3289:1943,dis
% load("PCMAC.mat");
% dataName="PCMAC";

% %RELATHE,4322:1427,dis
% load("RELATHE.mat");
% dataName="RELATHE";

% %gisette,5000:6000,con
% load("gisette.mat");
% dataName="gisette";

% %Leukemia1,5327:72,con
% load("Leukemia1.mat");
% dataName="Leukemia1";

% %9_Tumors,5726:60,con
% load("9_Tumors.mat");
% dataName="9_Tumors";

% %11_Tumors,12533:174,con
% load("11_Tumors.mat");
% dataName="11_Tumors";

% %Lung_Cancer,12600:203,con
% load("Lung_Cancer.mat");
% dataName="Lung_Cancer";

% %data_BLCA,12542:426,con
load("data_BLCA.mat");
dataName="data_BLCA";

% %data_BRCA,12657:1218,con
% load("data_BRCA.mat");
% dataName="data_BRCA";

% %data_LGG,13987:530,con
% load("data_LGG.mat");
% dataName="data_LGG";
  
% %data_LUAD,13431:574,con
% load("data_LUAD.mat");
% dataName="data_LUAD";

% %data_LUSC,13400:553,con.322
% load("data_LUSC.mat");
% dataName="data_LUSC";

%BLCA_test,12542:80+8,con
% load("BLCA_test.mat");
% dataName="BLCA_test";

data = X;
% data = data + 2;
label = Y;
% label = label + 2;
% %数据归一化
% data_min=min(data);
% data_max=max(data);
% data=(data-data_min+0.001)./(data_max-data_min);

%超参数设置

% % 设置随机种子
% random_seed=42;
% rng(random_seed);

%开始迭代
repeat=1;
foldNum=10;
acc_repeat=[];
numMB_repeat=[];
feature_repeat=[];
time_repeat=[];
for k = 1:repeat
    indices=crossvalind('Kfold',label,foldNum);
    acc_fold=[];
    numMB_fold=[];
    feature_fold=[];
    time_fold=[];
    for fold = 1:foldNum %十折交叉验证
        disp("进入交叉验证");
        boo=indices==fold;
        testX = data(indices==fold,:);
        testY = label(indices==fold,:); 
        trainX = data(indices~=fold,:);
        trainY = label(indices~=fold,:);
        set=[trainY,trainX];
        target=1;
        alpha=0.01;
        ns=max(set);
        [samples,p] = size(set);
        maxK=3;
        %tic;
        % 算法
   %    MMMB
   %    [MB,test,time]=MMMB_G2(set,target,alpha,ns,p,maxK);
   %    [MB,test,time]=MMMB_Z(set,target,alpha,samples,p,maxK);
   %    HITONMB
   %    [MB,test,time]=HITONMB_G2(set,target,alpha,ns,p,maxK);
   %    [MB,test,time]=HITONMB_Z(set,target,alpha,samples,p,maxK);
   %    PCMB
   %    [MB,test,time]=PCMB_G2(set,target,alpha,ns,p,maxK);
   %    [MB,test,time]=PCMB_Z(set,target,alpha,samples,p,maxK);
   %    STMB
   %    [MB,test,time]=STMB_G2(set,target,alpha,ns,p,maxK);
   %    [MB,test,time]=STMB_Z(set,target,alpha,samples,p,maxK);
   %    BAMB 
   %    [MB,test,time]=BAMB_G2(set,target,alpha,ns,p,maxK);
   %    [MB,test,time]=BAMB_Z(set,target,alpha,samples,p,maxK);
   %    EEMB
   %    [MB,test,time]=EEMB_G2(set,target,alpha,ns,p,maxK);
   %    [MB,test,time]=EEMB_Z(set,target,alpha,samples,p,maxK);
   %    EAMB
   %    [MB,test,time]=EAMB_G2(set,target,alpha,ns,p,maxK);
   %    [MB,test,time]=EAMB_Z(set,target,alpha,samples,p,maxK);
   %    CFS_MI
   %    [MB,test,time]=CFS_MI_G2(set,target,alpha,ns,p,maxK);
   %    my
   %    [MB,test,time]=my_CFS_MI_G2(set,target,alpha,ns,p,maxK);
   %    mag
       [MB,test,time]=mag_G2(set,target,alpha,ns,p,maxK);
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %time = toc;
        time_fold=[time_fold,time];
        MB=MB-1;
        %显示所选特征
        disp(MB);
        numMB_fold = [numMB_fold,size(MB,2)];
        %选择KNN/SVM分类器
        [featureNum_choice,acc]=test_m(trainX,trainY,testX,testY,MB);
        %[featureNum_choice,acc]=test_svm(trainX,trainY,testX,testY,MB);
        %disp("测试集上的精度："+acc);
        acc_fold=[acc_fold,acc];
        feature_fold=[feature_fold,featureNum_choice];
    end
    acc_fold_mean=sum(acc_fold)./foldNum;
    %feature_fold_mean=mean(feature_fold);
    numMB_fold_mean=mean(numMB_fold);
    time_fold_mean=mean(time_fold);
    %disp("10折交叉验证测试集上的平均精度："+acc_fold_mean);
    acc_repeat=[acc_repeat,acc_fold_mean];
    numMB_repeat=[numMB_repeat,numMB_fold_mean];
    %feature_repeat=[feature_repeat,feature_fold_mean];
    time_repeat=[time_repeat,time_fold_mean];
end
acc_repeat_mean=sum(acc_repeat)./repeat;
numMB_repeat_mean=sum(numMB_repeat)/repeat;
feature_repeat_mean=mean(feature_repeat);
time_repeat_mean=mean(time_repeat);
disp("多次实验平均precision："+acc_repeat_mean);
disp("多次实验平均size："+numMB_repeat_mean);
%disp("多次实验平均size："+feature_repeat_mean);
disp("多次实验平均time："+time_repeat_mean);

% result=struct();
% result.acc_repeat_mean=acc_repeat_mean;
% result.feature_repeat_mean=feature_repeat_mean;
% result.time_repeat_mean=time_repeat_mean;
% result.acc_repeat=acc_repeat;
% result.feature_repeat=feature_repeat;
% result.time_repeat=time_repeat;
% dataName=dataName+"_result.mat";
% save(dataName,"result");
disp('*******************end****************');