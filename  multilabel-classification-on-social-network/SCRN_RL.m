function [pred_label,accuracy]= SCRN_RL(Net,IDX,Label,SF,maxiter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Iterative relational neighbor classifier using social context features (SCRN)
% using Relaxation labeling for collecive inference.
% Our evaluation scheme is based on "network cross-validation" scheme proposed in paper: 
% - Neville, Jennifer et al. "Correcting evaluation bias of relational classifiers with network cross validation."
% Knowledge and Information Systems (Jan 2011), 1–25.
%
% INPUT:
% Net: a network interaction matrix
% IDX: a strcture which contains the node indices for training, inference and testing 
% IDX contains: IDX.training, IDX.inference, IDX.testing
% Label: the labels for all nodes in the network
% SF: the social features for all nodes in the network, obtained by EdgeClustering method
% maxiter: the maximum number of iterations for SCRN
% 
% OUTPUT:
% pred_label: the predict labels for testing nodes
% pred_prob: the prediction probability of the labels for testing nodes
% Updated by Xi Wang
% 01/07/2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

numNodes = size(Net,1);
tr_IDX = IDX.training;
inference_IDX = IDX.inference;
test_IDX = IDX.testing;
tr_label = Label(tr_IDX,:);
test_label = Label(test_IDX,:);
inference_label = Label(inference_IDX,:);
SF_tr = SF(tr_IDX,:);
SF_inference = SF(inference_IDX,:);

[numtr,numf] = size(SF_tr); % the dimension of the node's social features
numClass = size(tr_label,2); % the number of class
num_inference = size(inference_label,1);


%% Initializing the feature vector for each class
Class_feature = zeros(numClass,numf);
for i=1:numClass
    member = (tr_label(:,i)==1);
    Class_feature(i,:) = sum(SF_tr(member,:),1);
end

%% Calculate the initial class propagation probability for testing node
method = 'hist';
Prob_cp = Calc_prob_cp(SF_inference, Class_feature, method);

%% SCRN start
iter = 1;
output = sparse(numNodes,numClass); % the output prob. for all nodes
output(tr_IDX,:) = tr_label;
old_label = sparse(num_inference,numClass);
pred_label = sparse(num_inference,numClass);

% The default parameters setting for relaxation labeling 
changed = num_inference; % the number test nodes changed
belta = 0.5;
alpha = 0.99;
P_total_old = sparse(num_inference,numClass);
while(iter <= maxiter && changed>=0) 
    P = zeros(num_inference,numClass);
    for i = 1:num_inference
        tmp = find(Net(inference_IDX(i),:)>0); % find the node's neighbors
        Node_sim = 'Degree'; % the method for calculating the similarity between node and its neighbors
        Sim = Calc_nodeW(Net,inference_IDX(i),tmp,Node_sim); 
        P(i,:) = Sim*(repmat(Prob_cp(i,:),length(tmp),1).*output(tmp,:));
        if(sum(P(i,:)))
            P(i,:) = P(i,:)/sum(P(i,:));
        end
        clear tmp;
    end 
    
    % Relaxation Labeling 
    P_total_new = belta*P + (1-belta)*P_total_old;
    tp = inference_IDX(find(sum(P_total_new,2)));
    tp2 = (sum(P_total_new,2)>0);
    P_total_new(tp2,:) = P_total_new(tp2,:)./repmat(sum(P_total_new(tp2,:),2),1,numClass);
    
    % Keep the predictions probability of the testing nodes according to the # of labels each test node could belong.
    method2 = 'all'; %'all','top'
    [output(tp,:),pred_label(tp2,:)] = Keep_prediction(P_total_new(tp2,:),Label(tp,:),method2);
    clear P2;
    changed = full(sum(sum(xor(pred_label,old_label))));
    fprintf('%d iteration, # of prediction changed = %d\n',iter,changed);
   
    %% Evaluation on the whole inference set
%     rate(1) = evaluation(pred_label,inference_label,numClass,'Macro');
%     rate(2) = evaluation(pred_label,inference_label,numClass,'Micro');
%     fprintf('%d iteration, Macro = %d; Micro = %d \n',iter,rate(1),rate(2));
    %% update the Class feature----weighted sum
    Class_feature = zeros(numClass,numf);
    for i = 1:numClass
        member = (output(:,i)>0);
        Class_feature(i,:) = output(member,i)'*SF(member,:);
        Class_feature(i,:) = Class_feature(i,:)./length(member);
    end
    %% (1) update the testing nodes' class propagation probability
    Prob_cp = Calc_prob_cp(SF_inference,Class_feature, method);
    old_label = pred_label;
    iter = iter +1;
    P_total_old = P_total_new;
    belta = belta*alpha;
end
%% Evaluate on the testing set
[C,I1] = intersect(inference_IDX,test_IDX);
pred_label = pred_label(I1,:);
accuracy(1) = evaluation(pred_label,test_label,numClass,'Macro');
accuracy(2) = evaluation(pred_label,test_label,numClass,'Micro');
fprintf('# %d iteration,Accuracy is %f : \n', iter, accuracy(1));
end

function Sim = Calc_nodeW(Net,IDX1,IDX2,method)
%Calculate the similarity (weights) between two nodes based on different methods.
% Methods include: Cosine, Pearson, Degree
switch method
    case 'Cosine'
        F1 = Net(IDX1,:);
        F2 = Net(IDX2,:);
        Sim = F1*F2;
        id = Sim>0;
        D1 = sqrt(sum(Net(IDX1,:).^2));
        D2 = sqrt(sum(Net(IDX2,:).^2,2));
        D = D1*D2';
        Sim(id) = Sim(id)./D(id);
    case 'Pearson' 
        N = sum(sum(Net))/2; 
        F1 = Net(IDX1,:);
        F2 = Net(IDX2,:);
        Sim = F1*F2;
        id = Sim>0;
        D1 = sum(Net(IDX1,:));
        D2 = sum(Net(IDX2,:),2);
        D = D1*D2';
        Sim(id) = N*Sim(id)./D(id);
    case 'Degree'
        Sim = Net(IDX1,IDX2)/sum(Net(IDX1,IDX2));         
end
end

function [output,pred_label] = Keep_prediction(P,test_label,method)
% keep the prediction prbability of test nodes' labels,
% assuming we know the number of classes each testing data associated with
% method:
% 'top': keep only top k predictions
% 'all': keep all predictions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[num_test,numClass] = size(P);
output = zeros(num_test,numClass);
pred_label = zeros(num_test,numClass);
N = sum(test_label,2); % the total number of class for each test nodes

switch method
    case 'top' 
        for i=1:num_test
            V = sort(full(P(i,:)),2,'descend');
            for j = 1:N(i)
                IDX = find(P(i,:)==V(j));
                output(i,IDX) = P(i,IDX);
                pred_label(i,IDX) = 1;
            end
            if(sum(output(i,:))~=0)
                output(i,:) = output(i,:)./sum(output(i,:)); %Normalize
            end
        end
    case 'all' % accept all the prediction
        for i=1:num_test
            [v,idx]= sort(full(P(i,:)),2,'descend');
            pred_label(i,idx(1:N(i))) = ones(1,N(i));
            output(i,:) = P(i,:)./sum(P(i,:)); %Normalize
        end
        
end
end

function Sim = Calc_prob_cp(testing,Class_feature, method)
% Calculate the propagation probability based on the node's social features
% and the class reference social features
[num_test,numf] = size(testing);
numClass = size(Class_feature,1);
Sim = zeros(num_test,numClass);
A = zeros(num_test,numf);
B = zeros(numClass,numf);

switch method
    case 'Cosine'
        % The points are normalized, centroids are not, so normalize them
        Anorm = sqrt(sum(testing.^2, 2));
        Bnorm = sqrt(sum(Class_feature.^2, 2));
        tp1 = find(Anorm>0);
        tp2 = find(Bnorm>0);
        A(tp1,:) = testing(tp1,:) ./ Anorm(tp1,ones(1,numf));
        B(tp2,:) = Class_feature(tp2,:) ./ Bnorm(tp2,ones(1,numf));
        Sim = A*B';
        % normalize
        tp = find(sum(Sim,2)>0);
        Sim(tp,:) = Sim(tp,:)./repmat(sum(Sim(tp,:),2),1,numClass);
    case 'Inner'
        Sim = testing*Class_feature';
        tp = find(sum(Sim,2)>0);
        Sim(tp,:) = Sim(tp,:)./repmat(sum(Sim(tp,:),2),1,numClass);
    case 'hist' %generalized histogram intersection kernel
        belta = 0.01; 
        for i = 1:numClass
            tmp = sum(min(testing.^belta,repmat(Class_feature(i,:).^belta,num_test,1)),2);
            Sim(:,i) = tmp';
        end
        tp = find(sum(Sim,2)>0);
        Sim(tp,:) = Sim(tp,:)./repmat(sum(Sim(tp,:),2),1,numClass);        
end
end

