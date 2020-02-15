function a3Test(imageName)
%a3Test gets the name of the image as an input and displays the
%probabilities of each number
%imageName  name of the image to load
    close all;  %close all figures
  
%Load the model 
    load model;    %load the model of the network
%Load the image and transform it
    number=imread(imageName);    %read the image file
    imshow(number);         %show the image
    movegui('northwest');
    number=im2double(number);   %transform the format from 8bit to double
    number = number';       %this is just to adapt to USPS database
    data=number(:);         %finally transform the matrix into a vector
%Compute the probabilities
    probabilities = computeProbabilities(model, data);
%Show the probabilities in a graph
  figure;
  numbers = [ 0 1 2 3 4 5 6 7 8 9];
  bar(numbers,probabilities);
  movegui('northeast');
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

function ret = computeProbabilities(model, data, wd_coefficient)
  % model.input_to_hid is a matrix of size (n_hid,256)
  % model.hid_to_class is a matrix of size (10,256)
  % data.inputs is a matrix of size (256,<number of data cases>)
  % data.outputs is a matrix of size (10,<number of data cases>)
    
  % first, do the forward pass, i.e. calculate a variety of relevant values
  hid_in = model.input_to_hid * data; % input to the hidden units, i.e. before the logistic. size: (n_hid,<number of data cases>)
  hid_out = logistic(hid_in); % output of the hidden units, i.e. after the logistic. size: (n_hid,<number of data cases>)
  class_in = model.hid_to_class * hid_out; % input to the components of the softmax. size: (10, <number of data cases>)
  class_normalizer = log_sum_exp_over_rows(class_in); % log(sum(exp)) is what we subtract to get normalized log class probabilities. size: (1,<number of data cases>)
  log_class_prob = class_in - repmat(class_normalizer, [size(class_in, 1), 1]); % log of probability of each class. size: (10, <number of data cases>)
  class_out = exp(log_class_prob); % probability of each class. Each column (i.e. each case) sums to 1. size: (10, <number of data cases>)
  ret = class_out;

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

