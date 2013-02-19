function test_pred_label = WvRN_RL2(Net,IDX,Label,maxiter)
% The weighted vote of reltional neighbor classifier (paper 'A simple
% relational classifier') with relexation labeling
% Our evaluation scheme is based on "network cross-validation" scheme proposed in paper:
% - Neville, Jennifer et al. "Correcting evaluation bias of relational classifiers with network cross validation."
% Knowledge and Information Systems (Jan 2011), 1–25.
%
% INPUT:
% Net: a network interaction matrix
% IDX: a strcture which contains the node indices for training, inference and testing
% IDX contains: IDX.training, IDX.inference, IDX.testing
% Label: the labels for all nodes in the network
% maxiter: the maximum number of iterations for wvRN

% Output:
% pred_label: the predicted labels for testing nodes
% Updated by Xi Wang
% 11/12/2012
% trid: the traing nodes id
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numNodes = size(Net,1);
tr_IDX = IDX.training;
inference_IDX = IDX.inference;
test_IDX = IDX.testing;
tr_label = Label(tr_IDX,:);
test_label = Label(test_IDX,:);
inference_label = Label(inference_IDX,:);
numInference = length(inference_IDX);
numClass = size(tr_label,2);

iter = 1;
Output = zeros(numNodes,numClass); % the predict probability of all inference nodes
Output(inference_IDX,:) = sparse(numInference,numClass);
Output(tr_IDX,:) = tr_label;
P_total_old = zeros(numInference,numClass);
count = full(sum(inference_label,2)); % the number of labels for each node

% the parameters setting for relaxation labeling
belta = 1;
alpha = 0.99;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while(iter <= maxiter)
    P = zeros(numInference,numClass); % the prob. distribution of the testing nodes
    for i = 1:numInference
        M = find(Net(inference_IDX(i),:));
        % the similarity between linked nodes is represented by link weights
        for j = 1:length(M)
            P(i,:) = P(i,:)+ Net(inference_IDX(i),M(j))*Output(M(j),:);
        end
    end    
    P = P./repmat(sum(Net(inference_IDX,:),2),1,numClass);
    
    % Relaxation Labeling
    P_total_new = belta*P + (1-belta)*P_total_old;
    method = 'all';  % keep all the prediction probability
    [Inf_Prob,pred_label] = Keep_prediction(P_total_new,inference_label,method);
    Output(inference_IDX,:) = Inf_Prob;
    iter = iter + 1;
    %% Evaluate on the whole inference set
%     accuracy(1) = evaluation(pred_label,inference_label,numClass,'Macro');
%     accuracy(2) = evaluation(pred_label,inference_label,numClass,'Micro');
%     fprintf('# %d iteration,Accuracy is Macro=%f, Micro=%f: \n', iter, accuracy(1),accuracy(2));
    P_total_old = P_total_new;
    belta = belta*alpha;
end
%% Evaluation on test set
[C,I1] = intersect(inference_IDX,test_IDX);
test_pred_label = pred_label(I1,:);
% rate(1) = evaluation(test_pred_label,test_label,numClass,'Macro');
% rate(2) = evaluation(test_pred_label,test_label,numClass,'Micro');
fprintf('For test set, Macro = %f,  Micro = %f : \n', rate(1),rate(2));

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
        [dSorted,dIndex] = sort(P,2,'descend');
        for j = 1:num_test
            idx = dIndex(j,1:N(j,:));
            pred_label(j,idx) = ones(1,N(j,:));   % predict the label of test node
        end
        output = P;
end
end