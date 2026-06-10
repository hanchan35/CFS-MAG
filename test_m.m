%用于计算测试集的精确度和最后选择的特征数
function [featureNum,Acc] = test_m(trainX,trainY,testX,testY,feature)
    pre = KNN(trainX(:,feature),trainY,testX(:,feature));
    %Acc=compute_bAcc(testY, pre);
    Acc=sum(pre==testY)/numel(testY);
    featureNum = sum(feature);

    %feature = find(result>theta);
    %model1 = fitcecoc(trainX(:,feature),trainY);
    %predic_label1 = predict(model1,testX(:,feature));
    %Acc=sum(predic_label1==testY)/numel(testY);
    %featureNum = sum(result);

end
