%SVM计算测试集的精确度和最后选择的特征数
function [featureNum,Acc] = test_svm(trainX,trainY,testX,testY,feature)

    %LibSVM版一类
    model = svmtrain(trainY, trainX, '-s 0 -t 2 -c 1');
    [predicted_label, accuracy, decision_values] = svmpredict(testY, testX, model);
    
    Acc = accuracy;
    featureNum = sum(feature);

end