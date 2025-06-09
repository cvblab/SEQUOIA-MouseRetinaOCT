%% Morphological Refinement of 3D Pseudo-Masks
% This script performs post-processing on segmented pseudo-masks stored in .mat files.
% The process includes morphological closing to smooth the mask,
% retaining the largest connected component per slice,
% followed by morphological opening to remove small artifacts.
%
% Note: After this preprocessing step, the 10 best-quality masks per volume
% were selected based on separate quality assessment criteria (not included here).

clear; clc;

%% Path Configuration (update these with your actual paths)
input_dir = '<input_directory>';   % e.g., 'data/pseudo_masks_raw/'
output_dir = '<output_directory>'; % e.g., 'data/pseudo_masks_refined/'

% Create output directory if it doesn't exist
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% List all .mat files in the input directory
mat_files = dir(fullfile(input_dir, '*.mat'));

% Define structuring elements for morphological operations
se_close = strel('disk', 10);  % For morphological closing
se_open  = strel('disk', 3);   % For morphological opening

%% Process Each Mask Volume
for idx = 1:length(mat_files)

    file_path = fullfile(input_dir, mat_files(idx).name);
    [~, sample_name, ~] = fileparts(mat_files(idx).name);

    % Load the 'masks' variable from the .mat file
    data = load(file_path);
    if ~isfield(data, 'masks')
        warning('File %s does not contain a variable named "masks". Skipping.', mat_files(idx).name);
        continue;
    end
    masks = data.masks;

    % Ensure the mask is 3D
    if ndims(masks) ~= 3
        warning('The mask in file %s is not 3D. Skipping.', mat_files(idx).name);
        continue;
    end

    [height, width, num_slices] = size(masks);
    refined_masks = false(height, width, num_slices);  % Initialize the refined mask volume

    %% Slice-by-Slice Morphological Processing
    for s = 1:num_slices
        current_slice = masks(:, :, s);

        % Apply morphological closing
        closed_slice = imclose(current_slice, se_close);

        % Identify connected components
        labeled = bwlabel(closed_slice);
        props = regionprops(labeled, 'Area');

        % Retain the largest region, if any
        if ~isempty(props)
            [~, max_idx] = max([props.Area]);
            largest_region = ismember(labeled, max_idx);
        else
            largest_region = false(height, width);
        end

        % Apply morphological opening
        final_slice = imopen(largest_region, se_open);
        refined_masks(:, :, s) = final_slice;
    end

    %% Save the refined mask volume to a new .mat file
    output_file = fullfile(output_dir, [sample_name, '.mat']);
    save(output_file, 'refined_masks');

    fprintf('Processed: %s\n', mat_files(idx).name);
end

disp('All files processed successfully.');
