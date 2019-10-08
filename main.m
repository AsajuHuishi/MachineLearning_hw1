clear
filename = '../wifi_localization.txt';
[data1,data2,data3,data4,data5,data6,data7,data8]=textread(filename,'%n%n%n%n%n%n%n%n');

X1 = [ones(size(data1,1),1),data1,data2,data3,data4,data5,data6,data7];
y = data8;
numtest = 25;
%train data
X_train = X1(1:2000-numtest,:);
y_train = y(1:2000-numtest,1);

%test data
X_test = X1(2000-numtest+1:2000,:);
y_test = y(2000-numtest+1:2000,:);

%% 三者选其一
t1 = clock;
%% linear kernel
% for i=1:10
% K1 = X_train*X_train';
% theta = X_train'*pinv(K1)*y_train;
% b = theta;
% end
%% noraml equ
for i=1:10000
[theta] = normalEqn(X_train,y_train);
b = theta;
end
%% regress
% for i=1:100000
% [b,bint,r,rint,stats] = regress(y_train,X_train);
% stats
% end
t2 = clock;
disp(['time',num2str(etime(t2,t1))])
%%
% model = svmtrain(y_train',X_train','-c 1 -g 0.07');
% [predict_label,accuracy]=svmpredict(y_test',X_test',model);


%% test
count = 0;

estimate = round(X_test*b);
groundtruth = y_test;

count = sum((estimate==groundtruth)==1);
numtest = size(estimate,1);
acc = count/numtest