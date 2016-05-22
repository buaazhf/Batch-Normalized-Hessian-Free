function nnet_demo_2
clear;clc;

seed = 1234;

randn('state', seed );
rand('twister', seed+1 );


%you will NEVER need more than a few hundred epochs unless you are doing
%something very wrong.  Here 'epoch' means parameter update, not 'pass over
%the training set'.
maxepoch = 100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%CURVES
%%%%%%%%%%%%%%%%%
%this dataset (by Ruslan Salakhutdinov) is available here: http://www.cs.toronto.edu/~jmartens/digs3pts_1.mat

tmp = load('digs3pts_1.mat');
indata = tmp.bdata';
%outdata = tmp.bdata;
intest = tmp.bdatatest';
%outtest = tmp.bdatatest;
clear tmp


%MNIST
%%%%%%%%%%%%%%%%%
%using the MNIST helper from UFLDL: http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset
%dataset: http://yann.lecun.com/exdb/mnist/

indata = loadMNISTImages('train-images.idx3-ubyte');
intest = loadMNISTImages('t10k-images.idx3-ubyte');


%STL-10
%%%%%%%%%%%%%%%%%
%this is the STL-10 dataset: https://cs.stanford.edu/~acoates/stl10/
% 
% indata = load('stl-train.mat','X');
% indata = indata';
% 
% intest = load('stl-test.mat','X');
% intest = intest';


%SVHN
%%%%%%%%%%%%%%%%%
%this is the Steet View House Number dataset: http://ufldl.stanford.edu/housenumbers/
% 
% indata = load('svhn-train_32x32.mat','X');
% indata = reshape(indata,32*32*3,size(indata,4));
% %indata = indata';
% 
% intest = load('svhn-test_32x32.mat','X');
% intest = reshape(intest,32*32*3,size(intest,4));
% %intest = intest';



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

perm = randperm(size(indata,2));
indata = indata( :, perm );

%it's an auto-encoder so output is input
outdata = indata;
outtest = intest;


runName = 'HFtestrun2';

runDesc = ['seed = ' num2str(seed) ', enter anything else you want to remember here' ];

%next try using autodamp = 0 for rho computation.  both for version 6 and
%versions with rho and cg-backtrack computed on the training set


layersizes = [400 200 100 50 25 6 25 50 100 200 400];
%Note that the code layer uses linear units
layertypes = {'logistic', 'logistic', 'logistic', 'logistic', 'logistic', 'linear', 'logistic', 'logistic', 'logistic', 'logistic', 'logistic', 'logistic'};

resumeFile = [];

paramsp = [];
Win = [];
bin = [];
Nin = [];
%[Win, bin] = loadPretrainedNet_curves;


numchunks = 4
numchunks_test = 4;


mattype = 'gn'; %Gauss-Newton.  The other choices probably won't work for whatever you're doing
%mattype = 'hess';
%mattype = 'empfish';

rms = 0;

hybridmode = 1;

%decay = 1.0;
decay = 0.95;

%jacket = 0;
%this enables Jacket mode for the GPU
jacket = 0;

errtype = 'L2'; %report the L2-norm error (in addition to the quantity actually being optimized, i.e. the log-likelihood)

%standard L_2 weight-decay:
weightcost = 2e-5
%weightcost = 0


nnet_train_2_BN( runName, runDesc, paramsp, Win, bin, Nin, resumeFile, maxepoch, indata, outdata, numchunks, intest, outtest, numchunks_test, layersizes, layertypes, mattype, rms, errtype, hybridmode, weightcost, decay, jacket);
