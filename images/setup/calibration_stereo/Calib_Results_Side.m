% Intrinsic and Extrinsic Camera Parameters
%
% This script file can be directly excecuted under Matlab to recover the camera intrinsic and extrinsic parameters.
% IMPORTANT: This file contains neither the structure of the calibration objects nor the image coordinates of the calibration points.
%            All those complementary variables are saved in the complete matlab data file Calib_Results.mat.
% For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/


%-- Focal length:
fc = [ 1560.971521974508278 ; 1602.907413605645615 ];

%-- Principal point:
cc = [ 855.725469143312353 ; 229.913940013275692 ];

%-- Skew coefficient:
alpha_c = 0.000000000000000;

%-- Distortion coefficients:
kc = [ -0.037936628802503 ; 0.109556066284161 ; -0.064985409031489 ; -0.001654191527303 ; 0.000000000000000 ];

%-- Focal length uncertainty:
fc_error = [ 121.219356454958131 ; 114.726636221712198 ];

%-- Principal point uncertainty:
cc_error = [ 125.900418033754192 ; 167.192797029638569 ];

%-- Skew coefficient uncertainty:
alpha_c_error = 0.000000000000000;

%-- Distortion coefficients uncertainty:
kc_error = [ 0.243754953206307 ; 0.466267813045446 ; 0.030984497076191 ; 0.035304044827000 ; 0.000000000000000 ];

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
omc_1 = [ 2.524437e+00 ; -2.204880e-01 ; -2.123802e-01 ];
Tc_1  = [ 2.610757e-01 ; 3.566708e-01 ; 1.967179e+00 ];
omc_error_1 = [ 6.217063e-02 ; 2.908767e-02 ; 7.699055e-02 ];
Tc_error_1  = [ 1.648832e-01 ; 2.194546e-01 ; 1.122561e-01 ];

%-- Image #2:
omc_2 = [ 2.554458e+00 ; -1.429996e-01 ; -3.968941e-01 ];
Tc_2  = [ 2.607156e-01 ; 3.132085e-01 ; 1.944265e+00 ];
omc_error_2 = [ 6.387852e-02 ; 2.892572e-02 ; 9.153763e-02 ];
Tc_error_2  = [ 1.630269e-01 ; 2.145691e-01 ; 1.051941e-01 ];

%-- Image #3:
omc_3 = [ 2.545094e+00 ; -8.654558e-02 ; 6.477472e-01 ];
Tc_3  = [ 2.562059e-01 ; 3.233817e-01 ; 1.887801e+00 ];
omc_error_3 = [ 8.430569e-02 ; 5.693162e-02 ; 9.557557e-02 ];
Tc_error_3  = [ 1.541723e-01 ; 2.028142e-01 ; 8.860124e-02 ];

%-- Image #4:
omc_4 = [ 2.045380e+00 ; -3.211142e-01 ; -1.453885e-01 ];
Tc_4  = [ 2.935862e-01 ; 4.018143e-01 ; 1.885351e+00 ];
omc_error_4 = [ 8.037438e-02 ; 4.537073e-02 ; 7.003393e-02 ];
Tc_error_4  = [ 1.599488e-01 ; 2.131300e-01 ; 1.067296e-01 ];

%-- Image #5:
omc_5 = [ 1.929025e+00 ; 7.658928e-01 ; -8.482497e-01 ];
Tc_5  = [ 2.708340e-01 ; 3.865041e-01 ; 1.959980e+00 ];
omc_error_5 = [ 7.944192e-02 ; 5.790992e-02 ; 7.629403e-02 ];
Tc_error_5  = [ 1.655255e-01 ; 2.194200e-01 ; 1.066509e-01 ];

%-- Image #6:
omc_6 = [ -1.908324e+00 ; -2.035044e+00 ; 1.362344e+00 ];
Tc_6  = [ 3.425514e-01 ; 2.030428e-01 ; 1.972266e+00 ];
omc_error_6 = [ 7.230350e-02 ; 6.683129e-02 ; 1.210254e-01 ];
Tc_error_6  = [ 1.641516e-01 ; 2.125858e-01 ; 9.175540e-02 ];

%-- Image #7:
omc_7 = [ 1.890951e+00 ; 1.855777e+00 ; -7.479199e-01 ];
Tc_7  = [ 3.113361e-01 ; 1.973621e-01 ; 2.038621e+00 ];
omc_error_7 = [ 5.248287e-02 ; 5.430622e-02 ; 7.993695e-02 ];
Tc_error_7  = [ 1.675420e-01 ; 2.208066e-01 ; 1.152894e-01 ];

%-- Image #8:
omc_8 = [ 2.234809e+00 ; -2.938152e-01 ; -5.665746e-01 ];
Tc_8  = [ 2.977942e-01 ; 3.250824e-01 ; 1.806137e+00 ];
omc_error_8 = [ 8.395212e-02 ; 3.770163e-02 ; 8.292638e-02 ];
Tc_error_8  = [ 1.527080e-01 ; 1.994290e-01 ; 8.865738e-02 ];

%-- Image #9:
omc_9 = [ 2.214225e+00 ; -2.922348e-01 ; -6.143594e-01 ];
Tc_9  = [ 2.786854e-01 ; 3.259564e-01 ; 1.812851e+00 ];
omc_error_9 = [ 8.636292e-02 ; 3.890706e-02 ; 8.322895e-02 ];
Tc_error_9  = [ 1.530629e-01 ; 1.995072e-01 ; 8.705286e-02 ];

%-- Image #10:
omc_10 = [ 2.470671e+00 ; -3.530211e-02 ; -5.971092e-01 ];
Tc_10  = [ 2.532849e-01 ; 2.875024e-01 ; 1.732582e+00 ];
omc_error_10 = [ 7.292036e-02 ; 3.475202e-02 ; 9.447007e-02 ];
Tc_error_10  = [ 1.457603e-01 ; 1.905056e-01 ; 8.669443e-02 ];

%-- Image #11:
omc_11 = [ 2.127360e+00 ; 1.547671e+00 ; -1.180588e+00 ];
Tc_11  = [ 3.957520e-01 ; 1.703395e-01 ; 1.847704e+00 ];
omc_error_11 = [ 5.038493e-02 ; 7.187152e-02 ; 1.064089e-01 ];
Tc_error_11  = [ 1.542541e-01 ; 2.004422e-01 ; 9.485638e-02 ];

%-- Image #12:
omc_12 = [ 1.858093e+00 ; 2.135112e+00 ; -1.180261e+00 ];
Tc_12  = [ 3.211576e-01 ; 1.498568e-01 ; 1.984594e+00 ];
omc_error_12 = [ 3.504183e-02 ; 7.546736e-02 ; 1.119639e-01 ];
Tc_error_12  = [ 1.642620e-01 ; 2.131041e-01 ; 1.014316e-01 ];

%-- Image #13:
omc_13 = [ NaN ; NaN ; NaN ];
Tc_13  = [ NaN ; NaN ; NaN ];
omc_error_13 = [ NaN ; NaN ; NaN ];
Tc_error_13  = [ NaN ; NaN ; NaN ];

%-- Image #14:
omc_14 = [ -1.863386e-01 ; -2.458010e+00 ; 1.245264e+00 ];
Tc_14  = [ 4.782179e-01 ; 1.995337e-01 ; 1.909734e+00 ];
omc_error_14 = [ 4.922511e-02 ; 8.155257e-02 ; 9.686814e-02 ];
Tc_error_14  = [ 1.602852e-01 ; 2.092929e-01 ; 1.029419e-01 ];

%-- Image #15:
omc_15 = [ -6.876150e-01 ; 2.454182e+00 ; -1.141009e+00 ];
Tc_15  = [ 4.657009e-01 ; 2.734565e-01 ; 2.007546e+00 ];
omc_error_15 = [ 6.992954e-02 ; 7.779279e-02 ; 1.081766e-01 ];
Tc_error_15  = [ 1.615029e-01 ; 2.173533e-01 ; 9.629284e-02 ];

%-- Image #16:
omc_16 = [ 1.186325e+00 ; -1.680830e+00 ; 3.770376e-01 ];
Tc_16  = [ 5.273594e-01 ; 2.640652e-01 ; 1.741456e+00 ];
omc_error_16 = [ 6.835725e-02 ; 7.082478e-02 ; 7.577645e-02 ];
Tc_error_16  = [ 1.488605e-01 ; 1.938743e-01 ; 9.052985e-02 ];

%-- Image #17:
omc_17 = [ 1.255417e+00 ; -1.904892e+00 ; 2.761024e-01 ];
Tc_17  = [ 4.418094e-01 ; 2.454913e-01 ; 1.591442e+00 ];
omc_error_17 = [ 6.376371e-02 ; 6.614054e-02 ; 7.989278e-02 ];
Tc_error_17  = [ 1.355461e-01 ; 1.761590e-01 ; 7.882432e-02 ];

%-- Image #18:
omc_18 = [ 2.096535e+00 ; -5.918824e-01 ; -3.892203e-01 ];
Tc_18  = [ 2.804576e-01 ; 3.207460e-01 ; 1.618123e+00 ];
omc_error_18 = [ 8.457188e-02 ; 3.867812e-02 ; 7.787707e-02 ];
Tc_error_18  = [ 1.375160e-01 ; 1.800170e-01 ; 8.084623e-02 ];

%-- Image #19:
omc_19 = [ 2.347034e+00 ; 7.874228e-01 ; -1.165325e+00 ];
Tc_19  = [ 2.683944e-01 ; 2.685633e-01 ; 1.734710e+00 ];
omc_error_19 = [ 7.229700e-02 ; 6.786554e-02 ; 1.032352e-01 ];
Tc_error_19  = [ 1.455971e-01 ; 1.885681e-01 ; 7.756352e-02 ];

%-- Image #20:
omc_20 = [ -6.310852e-02 ; 2.583862e+00 ; -7.673285e-01 ];
Tc_20  = [ 4.573555e-01 ; 2.901695e-01 ; 2.143196e+00 ];
omc_error_20 = [ 5.518734e-02 ; 7.829262e-02 ; 1.097929e-01 ];
Tc_error_20  = [ 1.731463e-01 ; 2.315727e-01 ; 1.016371e-01 ];

%-- Image #21:
omc_21 = [ -8.353544e-01 ; -2.383108e+00 ; 1.405552e+00 ];
Tc_21  = [ 4.696391e-01 ; 1.703553e-01 ; 2.104445e+00 ];
omc_error_21 = [ 6.203579e-02 ; 7.969125e-02 ; 1.067210e-01 ];
Tc_error_21  = [ 1.754999e-01 ; 2.277541e-01 ; 1.076215e-01 ];

%-- Image #22:
omc_22 = [ -8.677689e-02 ; -2.303679e+00 ; 6.730970e-01 ];
Tc_22  = [ 4.598750e-01 ; 1.154814e-01 ; 1.864175e+00 ];
omc_error_22 = [ 4.476129e-02 ; 9.278060e-02 ; 9.447258e-02 ];
Tc_error_22  = [ 1.542342e-01 ; 2.000043e-01 ; 8.524712e-02 ];

%-- Image #23:
omc_23 = [ 1.180925e+00 ; -1.990808e+00 ; 5.064630e-01 ];
Tc_23  = [ 5.596804e-01 ; 2.925927e-01 ; 1.708112e+00 ];
omc_error_23 = [ 5.174929e-02 ; 6.817529e-02 ; 7.749184e-02 ];
Tc_error_23  = [ 1.468600e-01 ; 1.929074e-01 ; 9.386813e-02 ];

%-- Image #24:
omc_24 = [ 1.776088e+00 ; -1.820404e+00 ; 2.561659e-01 ];
Tc_24  = [ 4.169446e-01 ; 3.726707e-01 ; 1.868308e+00 ];
omc_error_24 = [ 4.711849e-02 ; 4.740910e-02 ; 8.586137e-02 ];
Tc_error_24  = [ 1.593624e-01 ; 2.111675e-01 ; 1.033244e-01 ];

%-- Image #25:
omc_25 = [ -1.157999e+00 ; -2.337960e+00 ; 1.212814e+00 ];
Tc_25  = [ 4.120522e-01 ; 2.014002e-01 ; 1.743727e+00 ];
omc_error_25 = [ 5.428492e-02 ; 7.880096e-02 ; 1.063949e-01 ];
Tc_error_25  = [ 1.466166e-01 ; 1.901931e-01 ; 8.420836e-02 ];

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

