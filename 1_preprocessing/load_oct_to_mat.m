% =============================== %
%   OCT Volume Preprocessing      %
% =============================== %

% Define the path to the OCT file (replace with your actual file path)
oct_file = 'C:\path\to\your\oct\file\sample_volume.oct';

% Open the OCT file, extract the intensity volume, and close the file handle
handle = OCTFileOpen(oct_file);
data = OCTFileGetIntensity(handle);
OCTFileClose(handle);

% Crop the volume: remove the top 28 rows and bottom 20 rows for cleaner visualization
data = data(29:end-20, :, :);

% Apply 3D Gaussian smoothing
% Sigma values: [1 1 2] --> light smoothing in X/Y, stronger along depth (Z)
data = imgaussfilt3(data, [1 1 2]);

% Normalize the volume to the [0, 1] range using min-max normalization
data = mat2gray(data);

% Save the processed volume to a .mat file
save('processed_volume.mat', 'data');



