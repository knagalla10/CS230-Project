function GenerateData(SNR, maxNoise, pct_pinkNoise, N_data, pct_dev, pct_test, varargin)

scaling_factors = [10^-9, 10^6, 1, 10^-9, 10^-9, 10^6, 1, 10^-9, 10^-9, 10^6, 1, 10^-6, 10^-6, 10^-12, 10^-12];
input_ranges = cell2mat(varargin{1:end});
[params, min_values, max_values, tp] = InitializeData(input_ranges, N_data);

parpool
S = zeros(length(tp(:, 1)), size(params, 2));

for i = 1:N_data
    % generate normalized response signal
    sys_props = params(:, i) .* scaling_factors';
    S_model = ThermalResponse(sys_props, tp*10^-12);

    % add random noise
    randNoise = pinknoise(length(tp), 1);
    randNoise = 2 * maxNoise * (randNoise - min(randNoise)) ./ (max(randNoise) - min(randNoise)) - maxNoise;
    S_noise = S_model + (pct_pinkNoise/100)*randNoise; % add pink noise
    S_noise = awgn(S_noise, SNR, 'measured', 1, 'linear') / (1-pct_pinkNoise/100); % add white noise

    % renormalize
    S(:, i) = (S_noise - min(S_noise)) ./ (max(S_noise) - min(S_noise));
end

% normalize parameter values between -1 to 1 for all combinations
normParams = 2 * ((params - repmat(min_values, 1, N_data)) ./ repmat(max_values-min_values, 1, N_data)) - 1;

% database of parameter combinations + simulated response signals (organized as input-output vector pairs)
inputVectors = [normParams(1:4, :); normParams(6, :); normParams(8, :); normParams(10:end, :); S];
outputVectors = [normParams(5, :); normParams(7, :); normParams(9, :)];

% randomly select examples for training, dev, and test sets
N_test = floor((pct_test/100) * N_data);
N_dev = floor((pct_dev/100) * N_data);
N_train_dev = N_data - N_test;
N_train = N_train_dev - N_dev;

data_IDs = randperm(N_data, N_data);
train_IDs = data_IDs <= N_train;
dev_IDs = data_IDs > N_train & data_IDs <= N_train_dev;
test_IDs = data_IDs > N_train_dev;

% training set examples
X_train = inputVectors(:, train_IDs);
Y_train = outputVectors(:, train_IDs);

% dev set examples
X_dev = inputVectors(:, dev_IDs);
Y_dev = outputVectors(:, dev_IDs);

% test set examples
X_test = inputVectors(:, test_IDs);
Y_test = outputVectors(:, test_IDs);

% parameter sampling ranges
X_ranges = [min_values(1:4, 1), max_values(1:4, 1); ...
    min_values(6, 1), max_values(6, 1); ...
    min_values(8, 1), max_values(8, 1); ...
    min_values(10:end, 1), max_values(10:end, 1)];
Y_ranges = [min_values(5, 1), max_values(5, 1); ...
    min_values(7, 1), max_values(7, 1); ...
    min_values(9, 1), max_values(9, 1)];

cd ..
mkdir data
cd data
mkdir training_data
mkdir dev_data
mkdir test_data
mkdir thermal_params

% save training data
writematrix(X_train, 'training_data/X_train.csv');
writematrix(Y_train, 'training_data/Y_train.csv');

% save dev data
writematrix(X_dev, 'dev_data/X_dev.csv');
writematrix(Y_dev, 'dev_data/Y_dev.csv');

% save test data
writematrix(X_test, 'test_data/X_test.csv');
writematrix(Y_test, 'test_data/Y_test.csv');

% save parameter ranges and delay time vector
writematrix(X_ranges, 'thermal_params/X_ranges.csv');
writematrix(Y_ranges, 'thermal_params/Y_ranges.csv');
writematrix(tp, 'thermal_params/timepoints.csv');

end