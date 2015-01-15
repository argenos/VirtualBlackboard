% Intrinsic and Extrinsic Camera Parameters
%
% This script file can be directly excecuted under Matlab to recover the camera intrinsic and extrinsic parameters.
% IMPORTANT: This file contains neither the structure of the calibration objects nor the image coordinates of the calibration points.
%            All those complementary variables are saved in the complete matlab data file Calib_Results.mat.
% For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/


%-- Focal length:
fc = [ 1668.925290652946387 ; 1639.016844913262048 ];

%-- Principal point:
cc = [ 1415.007376166849099 ; 686.436336786753031 ];

%-- Skew coefficient:
alpha_c = 0.000000000000000;

%-- Distortion coefficients:
kc = [ -0.009553316840761 ; 0.072902860977246 ; 0.014622088800406 ; 0.068661940342471 ; 0.000000000000000 ];

%-- Focal length uncertainty:
fc_error = [ 97.127282572828150 ; 67.414096981996565 ];

%-- Principal point uncertainty:
cc_error = [ 151.720081635768281 ; 122.653290496434025 ];

%-- Skew coefficient uncertainty:
alpha_c_error = 0.000000000000000;

%-- Distortion coefficients uncertainty:
kc_error = [ 0.100762182444469 ; 0.085825318764122 ; 0.016591676097067 ; 0.030673941779190 ; 0.000000000000000 ];

%-- Image size:
nx = 1920;
ny = 1080;


%-- Various other variables (may be ignored if you do not use the Matlab Calibration Toolbox):
%-- Those variables are used to control which intrinsic parameters should be optimized

n_ima = 27;						% Number of calibration images
est_fc = [ 1 ; 1 ];					% Estimation indicator of the two focal variables
est_aspect_ratio = 1;				% Estimation indicator of the aspect ratio fc(2)/fc(1)
center_optim = 1;					% Estimation indicator of the principal point
est_alpha = 0;						% Estimation indicator of the skew coefficient
est_dist = [ 1 ; 1 ; 1 ; 1 ; 0 ];	% Estimation indicator of the distortion coefficients


%-- Extrinsic parameters:
%-- The rotation (omc_kk) and the translation (Tc_kk) vectors for every calibration image and their uncertainties

%-- Image #1:
omc_1 = [ -1.406562e+00 ; 1.401219e+00 ; -8.701979e-01 ];
Tc_1  = [ -5.294916e-01 ; -6.219378e-02 ; 1.269211e+00 ];
omc_error_1 = [ 6.634334e-02 ; 5.208134e-02 ; 6.821815e-02 ];
Tc_error_1  = [ 1.365408e-01 ; 1.045121e-01 ; 3.895678e-02 ];

%-- Image #2:
omc_2 = [ -1.491757e+00 ; 1.216275e+00 ; -8.705031e-01 ];
Tc_2  = [ -4.955858e-01 ; -4.680872e-02 ; 1.232837e+00 ];
omc_error_2 = [ 6.840074e-02 ; 5.570088e-02 ; 6.848457e-02 ];
Tc_error_2  = [ 1.310169e-01 ; 1.011544e-01 ; 3.643264e-02 ];

%-- Image #3:
omc_3 = [ -9.230876e-01 ; 1.797958e+00 ; -1.401508e+00 ];
Tc_3  = [ -4.803627e-01 ; -4.696874e-02 ; 1.233701e+00 ];
omc_error_3 = [ 8.706984e-02 ; 5.349108e-02 ; 8.133539e-02 ];
Tc_error_3  = [ 1.275592e-01 ; 1.002500e-01 ; 4.161354e-02 ];

%-- Image #4:
omc_4 = [ -1.572789e+00 ; 1.834319e+00 ; -4.732553e-01 ];
Tc_4  = [ -4.739491e-01 ; -6.464314e-02 ; 1.331597e+00 ];
omc_error_4 = [ 5.268887e-02 ; 5.030436e-02 ; 7.263249e-02 ];
Tc_error_4  = [ 1.390646e-01 ; 1.075570e-01 ; 4.095266e-02 ];

%-- Image #5:
omc_5 = [ -2.536442e+00 ; 8.218538e-01 ; -8.189991e-01 ];
Tc_5  = [ -5.338409e-01 ; -5.360455e-02 ; 1.299199e+00 ];
omc_error_5 = [ 6.272896e-02 ; 3.550741e-02 ; 1.052840e-01 ];
Tc_error_5  = [ 1.397257e-01 ; 1.064491e-01 ; 4.299543e-02 ];

%-- Image #6:
omc_6 = [ -2.256885e+00 ; -7.442584e-01 ; -1.597104e+00 ];
Tc_6  = [ -4.945557e-01 ; -1.325866e-01 ; 1.108433e+00 ];
omc_error_6 = [ 7.778518e-02 ; 6.118086e-02 ; 9.191492e-02 ];
Tc_error_6  = [ 1.175624e-01 ; 9.324393e-02 ; 3.430019e-02 ];

%-- Image #7:
omc_7 = [ 2.587954e+00 ; 3.901948e-02 ; 1.793628e+00 ];
Tc_7  = [ -5.674275e-01 ; -1.004104e-01 ; 1.086361e+00 ];
omc_error_7 = [ 7.727110e-02 ; 5.660402e-02 ; 9.449483e-02 ];
Tc_error_7  = [ 1.235137e-01 ; 9.232631e-02 ; 3.949683e-02 ];

%-- Image #8:
omc_8 = [ -1.620679e+00 ; 1.322923e+00 ; -4.829653e-01 ];
Tc_8  = [ -3.683485e-01 ; -9.060985e-02 ; 1.276834e+00 ];
omc_error_8 = [ 5.801842e-02 ; 5.961525e-02 ; 7.564256e-02 ];
Tc_error_8  = [ 1.280209e-01 ; 1.020458e-01 ; 3.378119e-02 ];

%-- Image #9:
omc_9 = [ -1.631020e+00 ; 1.285218e+00 ; -5.252033e-01 ];
Tc_9  = [ -3.569193e-01 ; -7.859757e-02 ; 1.263344e+00 ];
omc_error_9 = [ 5.992147e-02 ; 5.928194e-02 ; 7.549232e-02 ];
Tc_error_9  = [ 1.263403e-01 ; 1.007199e-01 ; 3.338011e-02 ];

%-- Image #10:
omc_10 = [ -1.548531e+00 ; 1.082784e+00 ; -8.609219e-01 ];
Tc_10  = [ -2.797752e-01 ; -5.019880e-02 ; 1.258517e+00 ];
omc_error_10 = [ 7.451581e-02 ; 6.205508e-02 ; 7.317613e-02 ];
Tc_error_10  = [ 1.231941e-01 ; 9.862494e-02 ; 3.358490e-02 ];

%-- Image #11:
omc_11 = [ -2.347662e+00 ; -1.354278e-01 ; -1.431213e+00 ];
Tc_11  = [ -3.784006e-01 ; -1.972699e-01 ; 1.107380e+00 ];
omc_error_11 = [ 7.770345e-02 ; 5.216084e-02 ; 8.994345e-02 ];
Tc_error_11  = [ 1.148541e-01 ; 9.051015e-02 ; 3.396727e-02 ];

%-- Image #12:
omc_12 = [ -2.403409e+00 ; -6.396102e-01 ; -1.706052e+00 ];
Tc_12  = [ -4.954432e-01 ; -1.125102e-01 ; 1.054547e+00 ];
omc_error_12 = [ 7.246273e-02 ; 6.295852e-02 ; 9.450647e-02 ];
Tc_error_12  = [ 1.155037e-01 ; 8.899020e-02 ; 3.404272e-02 ];

%-- Image #13:
omc_13 = [ NaN ; NaN ; NaN ];
Tc_13  = [ NaN ; NaN ; NaN ];
omc_error_13 = [ NaN ; NaN ; NaN ];
Tc_error_13  = [ NaN ; NaN ; NaN ];

%-- Image #14:
omc_14 = [ 1.582939e+00 ; 1.899440e+00 ; 1.092901e+00 ];
Tc_14  = [ -4.519553e-01 ; -2.769952e-01 ; 1.142981e+00 ];
omc_error_14 = [ 6.359967e-02 ; 5.013568e-02 ; 7.821404e-02 ];
Tc_error_14  = [ 1.224452e-01 ; 9.571539e-02 ; 3.598380e-02 ];

%-- Image #15:
omc_15 = [ 1.430881e+00 ; 1.306692e+00 ; 2.211732e-01 ];
Tc_15  = [ -5.855629e-01 ; -2.488121e-01 ; 1.141925e+00 ];
omc_error_15 = [ 4.830587e-02 ; 7.831087e-02 ; 7.745144e-02 ];
Tc_error_15  = [ 1.199955e-01 ; 9.444199e-02 ; 4.989243e-02 ];

%-- Image #16:
omc_16 = [ -2.101589e-01 ; 2.209926e+00 ; 5.922251e-01 ];
Tc_16  = [ -3.152451e-01 ; -3.301590e-01 ; 1.242153e+00 ];
omc_error_16 = [ 3.745263e-02 ; 7.236027e-02 ; 8.377038e-02 ];
Tc_error_16  = [ 1.242810e-01 ; 1.008713e-01 ; 3.691617e-02 ];

%-- Image #17:
omc_17 = [ -9.043776e-02 ; 1.957585e+00 ; 5.763666e-01 ];
Tc_17  = [ -1.590629e-01 ; -2.541526e-01 ; 1.252328e+00 ];
omc_error_17 = [ 4.802758e-02 ; 7.909093e-02 ; 7.506341e-02 ];
Tc_error_17  = [ 1.199840e-01 ; 9.749168e-02 ; 3.663312e-02 ];

%-- Image #18:
omc_18 = [ -1.404368e+00 ; 1.546182e+00 ; -2.971123e-01 ];
Tc_18  = [ -1.884195e-01 ; -8.604014e-02 ; 1.318017e+00 ];
omc_error_18 = [ 5.482561e-02 ; 6.526686e-02 ; 8.072778e-02 ];
Tc_error_18  = [ 1.253634e-01 ; 1.017465e-01 ; 3.450675e-02 ];

%-- Image #19:
omc_19 = [ -2.094892e+00 ; 3.493536e-01 ; -1.053057e+00 ];
Tc_19  = [ -2.936042e-01 ; -7.385984e-02 ; 1.230549e+00 ];
omc_error_19 = [ 7.965090e-02 ; 5.308295e-02 ; 8.450551e-02 ];
Tc_error_19  = [ 1.198759e-01 ; 9.695269e-02 ; 3.257704e-02 ];

%-- Image #20:
omc_20 = [ 1.591375e+00 ; 9.610989e-01 ; 7.219902e-01 ];
Tc_20  = [ -7.129174e-01 ; -2.348592e-01 ; 1.136921e+00 ];
omc_error_20 = [ 5.967148e-02 ; 7.158944e-02 ; 7.176664e-02 ];
Tc_error_20  = [ 1.282251e-01 ; 9.765101e-02 ; 5.220534e-02 ];

%-- Image #21:
omc_21 = [ 2.090233e+00 ; 1.687206e+00 ; 1.407820e+00 ];
Tc_21  = [ -6.248330e-01 ; -2.531198e-01 ; 1.053914e+00 ];
omc_error_21 = [ 7.860838e-02 ; 3.715117e-02 ; 9.150004e-02 ];
Tc_error_21  = [ 1.195518e-01 ; 9.483989e-02 ; 3.770667e-02 ];

%-- Image #22:
omc_22 = [ 9.613518e-01 ; 1.923508e+00 ; 1.495503e+00 ];
Tc_22  = [ -3.798211e-01 ; -2.617477e-01 ; 1.058152e+00 ];
omc_error_22 = [ 8.061053e-02 ; 6.074061e-02 ; 8.587087e-02 ];
Tc_error_22  = [ 1.100553e-01 ; 8.837376e-02 ; 3.371523e-02 ];

%-- Image #23:
omc_23 = [ 1.443384e-01 ; 2.064709e+00 ; 5.327614e-01 ];
Tc_23  = [ -2.955648e-01 ; -3.655197e-01 ; 1.286758e+00 ];
omc_error_23 = [ 4.315586e-02 ; 7.148554e-02 ; 7.250839e-02 ];
Tc_error_23  = [ 1.296669e-01 ; 1.030790e-01 ; 4.145072e-02 ];

%-- Image #24:
omc_24 = [ -1.803455e-01 ; 1.828400e+00 ; 1.396237e-01 ];
Tc_24  = [ -4.574026e-01 ; -2.028961e-01 ; 1.309531e+00 ];
omc_error_24 = [ 4.516314e-02 ; 7.326103e-02 ; 6.333082e-02 ];
Tc_error_24  = [ 1.375172e-01 ; 1.064218e-01 ; 3.947141e-02 ];

%-- Image #25:
omc_25 = [ 2.079171e+00 ; 1.482523e+00 ; 1.710929e+00 ];
Tc_25  = [ -2.770985e-01 ; -2.146886e-01 ; 1.176201e+00 ];
omc_error_25 = [ 9.724453e-02 ; 4.228742e-02 ; 1.040303e-01 ];
Tc_error_25  = [ 1.159996e-01 ; 9.359495e-02 ; 3.532207e-02 ];

%-- Image #26:
omc_26 = [ NaN ; NaN ; NaN ];
Tc_26  = [ NaN ; NaN ; NaN ];
omc_error_26 = [ NaN ; NaN ; NaN ];
Tc_error_26  = [ NaN ; NaN ; NaN ];

%-- Image #27:
omc_27 = [ NaN ; NaN ; NaN ];
Tc_27  = [ NaN ; NaN ; NaN ];
omc_error_27 = [ NaN ; NaN ; NaN ];
Tc_error_27  = [ NaN ; NaN ; NaN ];

