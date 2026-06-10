function   [mb,ntest,time]=mag_G2(Data,target,~,ns,p,k)
start=tic;
ntest=0;
mb=[];
time=0;

%**********************************第一阶段，初始化，计算SU(T,X)****************************************%
% 分离target和(注意index问题)
train_data=Data(:,mysetdiff(1:p,target));
%被删除特征
removed=4;
%removed=25;
train_data=train_data(:,mysetdiff(1:p-1,removed));
featureMatrix = train_data;
% 标签，这里就是target这列的数据
train_label=Data(:,target);
%T5-4
removed_label=Data(:,4);
% %T30-25
% removed_label=Data(:,25);

classColumn = train_label;
numFeatures = size(featureMatrix,2);
%classScore构造一个全零的矩阵有numFeatures行，一列
classScore = zeros(numFeatures,1);

for i = 1:numFeatures
    ntest=ntest+1;
    classScore(i) = SU(featureMatrix(:,i),classColumn);
        %%%%%%%%%%%%%%%%%%%%%%%输出SU查看%%%%%%%%%%%%%%%%%%%
%     if i<target
%          fprintf('\ntarget=%.0f, i=%.0f, classScore(i)=%.4f\n',target,i,classScore(i));
%     else
%          fprintf('\ntarget=%.0f, i=%.0f, classScore(i)=%.4f\n',target,i+2,classScore(i));
%     end
end


% 排序，1--列的意思，desend降序，indexScore排序后元素在原矩阵中的位置
[classScore, indexScore] = sort(classScore,1,'descend');
%%%%%%%%%%%%%%%%%%%显示排序后的结果%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for i = 1:length(indexScore) 
%     if target < removed
%         if indexScore(i) < target
%             %disp(indexScore(i));
%             fprintf('\ni=%.0f, classScore(i)=%.4f\n',indexScore(i),classScore(i));
%         elseif indexScore(i)>=target && indexScore(i)<removed
%             tempDisp = indexScore(i)+1;
%             if tempDisp == removed
%                 tempDisp = tempDisp + 1;
%             end
%             %disp(tempDisp);
%             fprintf('\ni=%.0f, classScore(i)=%.4f\n',tempDisp,classScore(i));
%         elseif indexScore(i) >= removed
%             %disp(indexScore(i)+2);
%             fprintf('\ni=%.0f, classScore(i)=%.4f\n',indexScore(i)+2,classScore(i));
%         end
%     else
%         if indexScore(i) < removed
%             %disp(indexScore(i));
%             fprintf('\ni=%.0f, classScore(i)=%.4f\n',indexScore(i),classScore(i));
%         elseif indexScore(i)>=removed && indexScore(i)<target
%             tempDisp = indexScore(i)+1;
%             if tempDisp == target
%                 tempDisp = tempDisp + 1;
%             end
%             %disp(tempDisp);
%             fprintf('\ni=%.0f, classScore(i)=%.4f\n',tempDisp,classScore(i));
%         elseif indexScore(i) >= target
%             %disp(indexScore(i)+2);
%             fprintf('\ni=%.0f, classScore(i)=%.4f\n',indexScore(i)+2,classScore(i));
%         end
%     end
% end


%**********************************第二阶段，搜索PC****************************************%
% 循环开始
pc = [];
% thresholdHead = 0.12;
countHead = 0;
pointerHead = 2;
pc = myunion(pc,indexScore(1));

while classScore(pointerHead) > 0.05
    %计算XY之间的SU
    for i=1:length(pc)
        %查看数据
        XY0 = SU(featureMatrix(:,pc(i)),featureMatrix(:,indexScore(pointerHead)));
        tempPC=pc(i);tempIndex=indexScore(pointerHead);
        if target < removed
            if tempPC>=target && tempPC<removed
                tempPC=tempPC+1;
                if pc(i)==removed
                    tempPC=tempPC+1;
                end
            elseif tempPC>=removed
                tempPC=tempPC+2;
            end
        else
            if tempPC>=removed && tempPC<target
                tempPC=tempPC+1;
                if tempPC==target
                    tempPC=tempPC+1;
                end
            elseif tempPC>=target
                tempPC=tempPC+2;
            end
        end

        if target < removed
            if tempIndex>=target && tempIndex<removed
                tempIndex=tempIndex+1;
                if pc(i)==removed
                    tempIndex=tempIndex+1;
                end
            elseif tempIndex>=removed
                tempIndex=tempIndex+2;
            end
        else
            if tempIndex>=removed && tempIndex<target
                tempIndex=tempIndex+1;
                if tempIndex==target
                    tempIndex=tempIndex+1;
                end
            elseif tempIndex>=target
                tempIndex=tempIndex+2;
            end
        end

        %fprintf('\nX=%.0f, Y=%.0f, XY0=%.4f, YT=%.4f\n',tempPC,tempIndex,XY0,classScore(pointerHead));

        %记得统计test次数
        ntest=ntest+1;
        % 如果XY之间的SU比Y和T之间的SU要大，Y就可以删除了，这个证明了的
        if XY0 > classScore(pointerHead)
            %fprintf('\nX=%.0f, Y=%.0f\n',pc(i),indexScore(pointerHead));
            miRT = mi(removed_label,train_label);
            miRY = mi(removed_label,featureMatrix(:,indexScore(pointerHead)));
            %disp(miRT);disp(miRY);
            %fprintf('\nmiRT=%.4f, miRY=%.4f\n',miRT,miRY);
            if miRT < 0.1 || miRY < 0.1
                break;
            else
                pc = myunion(pc,indexScore(pointerHead));
            end
        elseif i==length(pc)
            pc = myunion(pc,indexScore(pointerHead));
        end
    end
    pointerHead = pointerHead + 1;
end
mb = yingshe2(pc,removed,target);

%**********************************第四阶段，搜索SP****************************************%
jXYZ = 0;
jXY0 = 0;
csp = [];
sp = [];
% countTail = 0;
% pointerTail = length(indexScore);

for i=1:length(pc)
    countTail = 0;
    pointerTail = length(indexScore);
    %while countTail <= 2 && pointerTail >= 1
    %while pointerTail >= 1 && classScore(pointerTail)<0.05   %默认值
    %while pointerTail >= 1 && classScore(pointerTail)<0.05
    while pointerTail >= 1
        % 需要先确保Y与Z有相关性，需要第三个阈值
        ntest = ntest + 1;
        if SU(featureMatrix(:,pc(i)),featureMatrix(:,indexScore(pointerTail))) > 0.13  %默认值,T5适用
        %if SU(featureMatrix(:,pc(i)),featureMatrix(:,indexScore(pointerTail))) > 0.04  %特殊值，T30适用
            %disp(indexScore(pointerTail));
            jXYZ = cmi(train_label,featureMatrix(:,indexScore(pointerTail)),featureMatrix(:,pc(i)));
            jXY0 = mi(train_label,featureMatrix(:,indexScore(pointerTail)));
            ntest=ntest+2;
            if jXYZ > jXY0
                countTail = 0;
                sp = myunion(sp,indexScore(pointerTail));
            else
                countTail = countTail + 1;
            end
        else
            countTail = countTail + 1;
        end
        pointerTail = pointerTail - 1;
    end
end

%**********************************第五阶段，拼接MB，返回结果****************************************%

mb = myunion(pc,sp);
mb = yingshe2(mb,removed,target);
disp("显示mb");
disp(mb);

time=toc(start);

end

function [score] = SU(firstVector,secondVector)
hX = h(firstVector);
hY = h(secondVector);
iXY = mi(firstVector,secondVector);
score = (2 * iXY) / (hX + hY);
end


function pc = yingshe2(pc, removed, target)
%YINGSHE2 此处显示有关此函数的摘要
%   此处显示详细说明
for i=1:length(pc)
    if target < removed
        if pc(i)>=target && pc(i)<removed
            pc(i)=pc(i)+1;
            if pc(i)==removed
                pc(i)=pc(i)+1;
            end
        elseif pc(i)>=removed
            pc(i)=pc(i)+2;
        end
    else
        if pc(i)>=removed && pc(i)<target
            pc(i)=pc(i)+1;
            if pc(i)==target
                pc(i)=pc(i)+1;
            end
        elseif pc(i)>=target
            pc(i)=pc(i)+2;
        end
    end
end
end

function P = P_A(A,a)
    count = 0;
    for i=1:length(A)
        if A(i)==a
            count = count + 1;
        end
    end
    P = count/length(A);
end


function P = P_A_B(A,a,B,b)
    count1 = 0;
    count2 = 0;
    for i=1:length(A)
        if B(i)==b 
            count1 = count1 + 1;
            if A(i)==a
                count2 = count2 + 1;
            end
        end
    end
    P = count2/count1;
end

function P = P_A_B_and_C(A,a,B,b,C,c)
    P = 0;
    count1 = 0;
    count2 = 0;
    for i=1:length(A)
        if B(i)==b && C(i)==c 
            count1 = count1 + 1;
            if A(i)==a
                count2 = count2 + 1;
            end
        end
    end
    %disp(count1);disp(count2);
    P = count2/count1;
    if isnan(P)
        P = 0;
    end
end
        










