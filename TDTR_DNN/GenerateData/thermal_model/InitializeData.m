function [data, min_values, max_values, tp] = InitializeData(input_ranges, N_data)

% simulated response time vector
N_points = 9750; % # of timepoints
t_inc = 2.04; % timepoint interval (ps)
t_start = -200; % measurement start timepoint (ps)
tp = (t_inc*(0:N_points-1) + t_start)'; % time vector

% parameter ranges
N_inputs = length(input_ranges(1, :)) / 2;
min_ids = 2*(1:N_inputs) - 1;
max_ids = 2*(1:N_inputs);
min_values = input_ranges(1, min_ids)';
max_values = input_ranges(1, max_ids)';

% shift up negative ranges
neg_IDs = min_values <= 0;
param_shifts = abs(min_values(neg_IDs, 1)) + 1;
min_values(neg_IDs, 1) = min_values(neg_IDs, 1) + param_shifts;
max_values(neg_IDs, 1) = max_values(neg_IDs, 1) + param_shifts;

% parameter combinations
data = zeros(N_inputs, N_data);
for i = 1:N_inputs
    pd = makedist("Loguniform", min_values(i, 1), max_values(i, 1));
    data(i, :) = random(pd, 1, N_data);
end

% shift back negative ranges
data(neg_IDs, :) = data(neg_IDs, :) - param_shifts;
min_values(neg_IDs, 1) = min_values(neg_IDs, 1) - param_shifts;
max_values(neg_IDs, 1) = max_values(neg_IDs, 1) - param_shifts;

end