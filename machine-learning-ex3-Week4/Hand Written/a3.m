function a3(wd_coefficient, n_hid, n_iters, learning_rate, momentum_multiplier, do_early_stopping, mini_batch_size)
%   wd_coefficient          Weight decay coefficient
%   n_hid                   Number of hidden units
%   n_iters                 Number of iterations
%   learing_rate
%   momentum_multiplier      
%   do_early_stopping
%   mini_batch_size
  warning('error', 'Octave:broadcast');
  if exist('page_output_immediately'), page_output_immediately(1); end
  more off;
  
  model = initial_model(n_hid);              %Initialize the weights
  from_data_file = load('data.mat');
  datas = from_data_file.data;
  n_training_cases = size(datas.training.inputs, 2);        %size(data.training.inputs)=(256 1000)
  %if n_iters ~= 0, test_gradient(model, datas.training, wd_coefficient); end    %test if the code for the gradient is ok

  % optimization
  theta = model_to_theta(model);
  momentum_speed = theta * 0;
  training_data_losses = [];
  validation_data_losses = [];
  if do_early_stopping,
    best_so_far.theta = -1; % this will be overwritten soon
    best_so_far.validation_loss = inf;
    best_so_far.after_n_iters = -1;
  end
  for optimization_iteration_i = 1:n_iters,
    fprintf('%d ',optimization_iteration_i);
    model = theta_to_model(theta);
    %Prepare the batcth for learning
    training_batch_start = mod((optimization_iteration_i-1) * mini_batch_size, n_training_cases)+1;
    training_batch.inputs = datas.training.inputs(:, training_batch_start : training_batch_start + mini_batch_size - 1);
    training_batch.outputs = datas.training.outputs(:, training_batch_start : training_batch_start + mini_batch_size - 1);
    %Compute the gradient and update the weights
    gradient = model_to_theta(d_loss_by_d_model(model, training_batch, wd_coefficient));
    momentum_speed = momentum_speed * momentum_multiplier - gradient;
    theta = theta + momentum_speed * learning_rate;

    model = theta_to_model(theta);
    training_data_losses = [training_data_losses, loss(model, datas.training, wd_coefficient)];
    validation_data_losses = [validation_data_losses, loss(model, datas.validation, wd_coefficient)];
    if do_early_stopping && validation_data_losses(end) < best_so_far.validation_loss,
      best_so_far.theta = theta; % this will be overwritten soon
      best_so_far.validation_loss = validation_data_losses(end);
      best_so_far.after_n_iters = optimization_iteration_i;
    end
    if mod(optimization_iteration_i, round(n_iters/10)) == 0,
      fprintf('\nAfter %d optimization iterations, training data loss is %f, and validation data loss is %f\n', optimization_iteration_i, training_data_losses(end), validation_data_losses(end));
    end
  end
  if n_iters ~= 0, test_gradient(model, datas.training, wd_coefficient); end % check again, this time with more typical parameters
  if do_early_stopping,
    fprintf('Early stopping: validation loss was lowest after %d iterations. We chose the model that we had then.\n', best_so_far.after_n_iters);
    theta = best_so_far.theta;
  end
  % the optimization is finished. Now do some reporting.
  model = theta_to_model(theta);
  clf;  %Clear current figure window
  hold on;
  plot(training_data_losses, 'b');
  plot(validation_data_losses, 'r');
  legend('training', 'validation');
  ylabel('loss');
  xlabel('iteration number');
  hold off;
  datas2 = {datas.training, datas.validation, datas.test};
  data_names = {'training', 'validation','test'};
  for data_i = 1:3,
    data = datas2{data_i};
    data_name = data_names{data_i};
    fprintf('\nThe loss on the %s data is %f\n', data_name, loss(model, data, wd_coefficient));
    if wd_coefficient~=0,
      fprintf('The classification loss (i.e. without weight decay) on the %s data is %f\n', data_name, loss(model, data, 0));
    end
    fprintf('The classification error rate on the %s data is %f\n', data_name, classification_performance(model, data));
  end
  save -mat7-binary model.mat model;
end

% test_gradient is a function used to test our implementation of the
% gradient calculation
function test_gradient(model, data, wd_coefficient)
  base_theta = model_to_theta(model);
  h = 1e-2;
  correctness_threshold = 1e-5;
  analytic_gradient = model_to_theta(d_loss_by_d_model(model, data, wd_coefficient));
  % Test the gradient not for every element of theta, because that's a lot of work. Test for only a few elements.
  for i = 1:100,
    test_index = mod(i * 1299721, size(base_theta,1)) + 1; % 1299721 is prime and thus ensures a somewhat random-like selection of indices
    analytic_here = analytic_gradient(test_index);
    theta_step = base_theta * 0;
    theta_step(test_index) = h;
    contribution_distances = [-4:-1, 1:4];
    contribution_weights = [1/280, -4/105, 1/5, -4/5, 4/5, -1/5, 4/105, -1/280];
    temp = 0;
    for contribution_index = 1:8,
      temp = temp + loss(theta_to_model(base_theta + theta_step * contribution_distances(contribution_index)), data, wd_coefficient) * contribution_weights(contribution_index);
    end
    fd_here = temp / h;
    diff = abs(analytic_here - fd_here);
    % fprintf('%d %e %e %e %e\n', test_index, base_theta(test_index), diff, fd_here, analytic_here);
    if diff < correctness_threshold, continue; end
    if diff / (abs(analytic_here) + abs(fd_here)) < correctness_threshold, continue; end
    error(sprintf('Theta element #%d, with value %e, has finite difference gradient %e but analytic gradient %e. That looks like an error.\n', test_index, base_theta(test_index), fd_here, analytic_here));
  end
 fprintf('Gradient test passed. That means that the gradient that your code computed is within 0.001%% of the gradient that the finite difference approximation computed, so the gradient calculation procedure is probably correct (not certainly, but probably).\n');
end

function ret = logistic(input)
  ret = 1 ./ (1 + exp(-input));
end

function ret = log_sum_exp_over_rows(a)
  % This computes log(sum(exp(a), 1)) in a numerically stable way
  maxs_small = max(a, [], 1);
  maxs_big = repmat(maxs_small, [size(a, 1), 1]);
  ret = log(sum(exp(a - maxs_big), 1)) + maxs_small;
end

function ret = loss(model, data, wd_coefficient)
  % model.input_to_hid is a matrix of size (n_hid,256)
  % model.hid_to_class is a matrix of size (10,256)
  % data.inputs is a matrix of size (256,<number of data cases>)
  % data.outputs is a matrix of size (10,<number of data cases>)
    
  % first, do the forward pass, i.e. calculate a variety of relevant values
  hid_in = model.input_to_hid * data.inputs; % input to the hidden units, i.e. before the logistic. size: (n_hid,<number of data cases>)
  hid_out = logistic(hid_in); % output of the hidden units, i.e. after the logistic. size: (n_hid,<number of data cases>)
  class_in = model.hid_to_class * hid_out; % input to the components of the softmax. size: (10, <number of data cases>)
  class_normalizer = log_sum_exp_over_rows(class_in); % log(sum(exp)) is what we subtract to get normalized log class probabilities. size: (1,<number of data cases>)
  log_class_prob = class_in - repmat(class_normalizer, [size(class_in, 1), 1]); % log of probability of each class. size: (10, <number of data cases>)
  class_out = exp(log_class_prob); % probability of each class. Each column (i.e. each case) sums to 1. size: (10, <number of data cases>)
  
  classification_loss = -mean(sum(log_class_prob .* data.outputs, 1)); % select the cross entropy right log class probability using that sum; then take the mean over all data cases.
  wd_loss = sum(model_to_theta(model).^2)/2*wd_coefficient; % very straightforward: E = 1/2 * lambda * theta^2
  ret = classification_loss + wd_loss;
end

function ret = d_loss_by_d_model(model, data, wd_coefficient)
  % model.input_to_hid is a matrix of size (n_hid,256)
  % model.hid_to_class is a matrix of size (10,n_hid)
  % data.inputs is a matrix of size (256,<number of data cases>)
  % data.outputs is a matrix of size (10,<number of data cases>)

  % The returned object is supposed to be exactly like parameter <model>, i.e. it has fields ret.input_to_hid and ret.hid_to_class. However, the contents of those matrices are gradients (d loss by d model parameter), instead of model parameters.
	 
  % This is the only function that you're expected to change. Right now, it just returns a lot of zeros, which is obviously not the correct output. Your job is to change that.
  ret.input_to_hid = model.input_to_hid * 0;
  ret.hid_to_class = model.hid_to_class * 0;
  
  % first, do the forward pass, i.e. calculate a variety of relevant values
  hid_in = model.input_to_hid * data.inputs; % input to the hidden units, i.e. before the logistic. size: (n_hid,<number of data cases>)
  hid_out = logistic(hid_in); % output of the hidden units, i.e. after the logistic. size: (n_hid,<number of data cases>)
  class_in = model.hid_to_class * hid_out; % input to the components of the softmax. size: (10, <number of data cases>)
  class_normalizer = log_sum_exp_over_rows(class_in); % log(sum(exp)) is what we subtract to get normalized log class probabilities. size: (1,<number of data cases>)
  log_class_prob = class_in - repmat(class_normalizer, [size(class_in, 1), 1]); % log of probability of each class. size: (10, <number of data cases>)
  class_out = exp(log_class_prob); % probability of each class. Each column (i.e. each case) sums to 1. size: (10, <number of data cases>)
  
  error_deriv = class_out - data.outputs;  %Error derivate w.r.t. zj size (10,<number of data cases>    
  hid_to_output_weights_gradient = [];
  for i=1:size(error_deriv,2),
      hid_to_output_weights_gradient(:,:,i) = hid_out(:,i)*error_deriv(:,i)'; %Gradient size must be(n_hid,10,<number of data cases>
  end
  hid_to_output_weights_gradient = mean(hid_to_output_weights_gradient,3);  %We need the mean 
  ret.hid_to_class = hid_to_output_weights_gradient';   %traspose to fit into the model
  
  backpropagate_error_deriv = model.hid_to_class'*error_deriv;   %size(n_hid,<number of data cases>) %Not sure about this
  input_to_hidden_weights_gradient = [];
  for i=1:size(backpropagate_error_deriv,2),
      input_to_hidden_weights_gradient(:,:,i) = data.inputs(:,i)*((1-hid_out(:,i)).*hid_out(:,i).*backpropagate_error_deriv(:,i))'; %Gradient size must be(n_hid,10,<number of data cases>
  end
  input_to_hidden_weights_gradient = mean (input_to_hidden_weights_gradient,3);
  ret.input_to_hid=input_to_hidden_weights_gradient';
  
  
  ret.input_to_hid = ret.input_to_hid + model.input_to_hid * wd_coefficient;        %ret.input_to_hid Size(256, n_hid) 
  ret.hid_to_class = ret.hid_to_class + model.hid_to_class * wd_coefficient;        %ret.hid_to_class Size(n_hid,10)
end

%Theta is a column vector that holds the weights
%Model contains two matrix (,) with the weights
function ret = theta_to_model(theta)
  n_hid = size(theta, 1) / (256+10);
  ret.input_to_hid = transpose(reshape(theta(1: 256*n_hid), 256, n_hid));
  ret.hid_to_class = reshape(theta(256 * n_hid + 1 : size(theta,1)), n_hid, 10).';
end

function ret = model_to_theta(model)
  input_to_hid_transpose = transpose(model.input_to_hid);
  hid_to_class_transpose = transpose(model.hid_to_class);
  ret = [input_to_hid_transpose(:); hid_to_class_transpose(:)];
end

function ret = initial_model(n_hid)
  n_params = (256+10) * n_hid;
  as_row_vector = cos(0:(n_params-1));
  ret = theta_to_model(as_row_vector(:) * 0.1); % We don't use random initialization, for this assignment. This way, everybody will get the same results.
end

function ret = classification_performance(model, data)
  % This returns the fraction of data cases that is incorrectly classified by the model.
  hid_in = model.input_to_hid * data.inputs; % input to the hidden units, i.e. before the logistic. size: <number of hidden units> by <number of data cases>
  hid_out = logistic(hid_in); % output of the hidden units, i.e. after the logistic. size: <number of hidden units> by <number of data cases>
  class_in = model.hid_to_class * hid_out; % input to the components of the softmax. size: <number of classes, i.e. 10> by <number of data cases>
  
  [dump, choices] = max(class_in); % choices is integer: the chosen class, plus 1.
  [dump, targets] = max(data.outputs); % targets is integer: the target class, plus 1.
  ret = mean(choices ~= targets);
end
