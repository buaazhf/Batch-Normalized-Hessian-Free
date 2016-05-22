function paramsp = nnet_train_2( runName, runDesc, paramsp, Win, bin, Nin, resumeFile, maxepoch, indata, outdata, numchunks, intest, outtest, numchunks_test, layersizes, layertypes, mattype, rms, errtype, hybridmode, weightcost, decay, jacket)
%
% Demo code for the paper "Deep Learning via Hessian-free Optimization" by James Martens.
%
% paramsp = nnet_train_2( runName, runDesc, paramsp, Win, bin, resumeFile, maxepoch, indata, outdata, numchunks, intest, outtest, numchunks_test, layersizes, layertypes, mattype, rms, errtype, hybridmode, weightcost, decay, jacket)
%
% IMPORTANT NOTES:  The most important variables to tweak are `initlambda' (easy) and
% `maxiters' (harder).  Also, if your particular application is still not working the next 
% most likely way to fix it is tweaking the variable `initcoeff' which controls
% overall magnitude of the initial random weights.  Please don't treat this code like a black-box,
% get a negative result, and then publish a paper about how the approach doesn't work :)  And if
% you are running into difficulties feel free to e-mail me at james.martens@gmail.com
%
% runName - name you give to the current run.  This is used for the
% log-file and the files which contain the current parameters that get
% saved every 10 epochs
%
% runDesc - notes to yourself about the current run (can be the empty string)
%
% paramsp - initial parameters in the form of a vector (can be []).  If
% this, or the arguments Win,bin are empty, the 'sparse initialization'
% technique is used
%
% Win, bin - initial parameters in their matrix forms (can be [])
% Nin - initial normalization parameters (gamma, beta)
%
% resumeFile - file used to resume a previous run from a file
%
% maxepoch - maximum number of 'epochs' (outer iteration of HF).  There is no termination condition
% for the optimizer and usually I just stop it by issuing a break command
%
% indata/outdata - input/output training data for the net (each case is a
% column).  Make sure that the training cases are randomly permuted when you invoke
% this function as it won't do it for you.
%
% numchunks - number of mini-batches used to partition the training set.
% During each epoch, a single mini-batch is used to compute the
% matrix-vector products, after which it gets cycled to the back of the
% last and is used again numchunk epochs later. Note that the gradient and 
% line-search are still computed using the full training set.  This of
% course is not practical for very large datasets, but in general you can
% afford to use a lot more data to compute the gradient than the
% matrix-vector products, since you only have to do the former once per iteration
% of the outer loop.
%
% intest/outtest -  test data
%
% numchunks_test - while the test set isn't used for matrix-vector
% products, you still may want to partition it so that it can be processed
% in pieces on the GPU instead of all at once.
%
% layersizes - the size of each hidden layer (input and output sizes aren't
% specified in this vector since they are determined by the dimension of
% the data arguements) 
%
% layertypes - a cell-array of strings that indicate the unit type in each
% layer.  can be 'logistic', 'tanh', 'linear' or 'softmax'.  I haven't
% thoroughly tested the softmax units.  Also, the output units can't be
% tanh because I haven't implemented that (even though it's easy).
% Consider that an exercise :)
%
% mattype - the type of curvature matrix to use.  can be 'gn' for
% Gauss-Newton, 'hess' for Hessian and 'empfish' for empirical Fisher.  You
% should probably only ever use 'gn' if you actually want the training to
% go well
%
% rms - by default we use the canonical error function for
% each output unit type.  e.g. square error for linear units and
% cross-entropy error for logistics.  Setting this to 1 (instead of 0) overrides 
% the default and forces it to use squared-error.  Note that even if what you
% care about is minimizing squared error it's sometimess still better
% to run on the optimizer with the canonical error
%
% errtype - in addition to displaying the objective function (log-likelihood) you may also
% want to keep track of another metric like squared error when you train
% deep auto-encoders.  This can be 'L2' for squared error, 'class' for
% classification error, or 'none' for nothing.  It should be easy enough to
% add your own type of error should you need one
%
% hybridmode - set this 1 unless you want compute the matrix-vector
% products using the whole training dataset instead of the mini-batches.
% Note that in this case they still serve a purpose since the mini-batches
% are only loaded onto the gpu 1 at a time.
%
% weightcost - the strength of the l_2 prior on the weights
%
% decay - the amount to decay the previous search direction for the
% purposes of initializing the next run of CG.  Should be 0.95
%
% jacket - set to 1 in order to use the Jacket computing library.  Will run
% on the CPU otherwise and hence be really slow.  You can easily port this code
% over to free and possibly slower GPU packages like GPUmat (in fact, I have some
% commented code which does just that (do a text search for "GPUmat version")




% arasu: replaced computeBV with compute GV, deleted the @function part for computeBV, removed compute FV, HV, and some testing stuff


disp( ['Starting run named: ' runName ]);

rec_constants = {'layersizes', 'rms', 'weightcost', 'hybridmode', 'autodamp', 'initlambda', 'drop', 'boost', 'numchunks', 'mattype', 'errtype', 'decay'};


autodamp = 1;

drop = 2/3;

boost = 1/drop;


%In addition to maxiters the variable below is something you should manually
%adjust.  It is quite problem specific.  Fortunately after only 1 'epoch'
%you can often tell if you've made a bad choice.  The value of rho should lie
%somewhere between 0.75 and 0.95.  I could automate this part but I'm lazy
%and my code isn't designed to make such automation a natural thing to add.  Also
%note that 'lambda' is being added to the normalized curvature matrix (i.e.
%divided by the number of cases) while in the ICML paper I was adding it to
%the unnormalized curvature matrix.  This doesn't make any real
%difference to the optimization, but does make it somewhat easier to guage
%lambda and set its initial value since it will be 'independent' of the
%number of training cases in each mini-batch
initlambda = 45.0;

if jacket
    mones = @ones;
    mzeros = @zeros;
    conv = @single;
    
    %GPUmat version:
    %mones = @(varargin) ones(varargin{:}, GPUsingle);
    %mzeros = @(varargin) zeros(varargin{:}, GPUsingle);
    %conv = @GPUsingle;
    
    %norm = @(x) sqrt(sum(x.*x));
    
    mrandn = @grandn;
else
    %use singles (this can make cpu code go faster):
    
    mones = @(varargin)ones(varargin{:}, 'single');
    mzeros = @(varargin)zeros(varargin{:}, 'single');
    %conv = @(x)x;
    conv = @single;
    
    
    %use doubles:
    %{
    mones = @ones;
    mzeros = @zeros;
    %conv = @(x)x;
    conv = @double;
    %}
    
    mrandn = @randn;
end

if hybridmode
    store = conv; %cache activities on the gpu

    %store = @single; %cache activities on the cpu
else
    store = @single;
end

% prepend, append input, output layer sizes to layersizes
layersizes = [size(indata,1) layersizes size(outdata,1)];
numlayers = size(layersizes,2) - 1;

[indims numcases] = size(indata); 
[tmp numtest] = size(intest);

if mod( numcases, numchunks ) ~= 0
    error( 'Number of chunks doesn''t divide number of training cases!' );
end

sizechunk = numcases/numchunks;
sizechunk_test = numtest/numchunks_test;


if numcases >= 512*64
    disp( 'jacket issues possible!' );
end

storeD = false
y = cell(numchunks, numlayers+1);
if storeD
    dEdy = cell(numchunks, numlayers+1);
    dEdx = cell(numchunks, numlayers);
end



function v = vec(A)
    v = A(:);
end

%{ for_bn: There are 2 extra parameters per neuron
%}
% psize = number of weights + number of biases + 2 normalization parameters for each layer
psize = layersizes(1,2:(numlayers+1))*layersizes(1,1:numlayers)' + sum(layersizes(2:(numlayers+1))) + 2*sum(layersizes(1:(numlayers)));

%pack all the parameters into a single vector for easy manipulation
function M = pack(W,b,N)
    
    M = mzeros( psize, 1 );
    
    cur = 0;
	
    for i = 1:numlayers
        M((cur+1):(cur + layersizes(i)*layersizes(i+1)), 1) = vec( W{i} );
        cur = cur + layersizes(i)*layersizes(i+1);
        
        M((cur+1):(cur + layersizes(i+1)), 1) = vec( b{i} );
        cur = cur + layersizes(i+1);
		
		%batch_norm
        M((cur+1):(cur + 2*layersizes(i)), 1) = vec( N{i} );
        cur = cur + 2*layersizes(i);
		
    end
    
end

%unpack parameters from a vector so they can be used in various neural-net
%computations
function [W,b,N] = unpack(M)

    W = cell( numlayers, 1 );
    b = cell( numlayers, 1 );
	N = cell( numlayers, 1 );
    
    cur = 0;
	
    for i = 1:numlayers
        W{i} = reshape( M((cur+1):(cur + layersizes(i)*layersizes(i+1)), 1), [layersizes(i+1) layersizes(i)] );
        cur = cur + layersizes(i)*layersizes(i+1);
        
        b{i} = reshape( M((cur+1):(cur + layersizes(i+1)), 1), [layersizes(i+1) 1] );
        cur = cur + layersizes(i+1);
		
		% batch_norm: one column for beta, another for gamma
        N{i} = reshape( M((cur+1):(cur + 2*layersizes(i)), 1), [layersizes(i) 2] );
        cur = cur + 2*layersizes(i);
		
    end
    
end

%compute the vector-product with the Gauss-Newton matrix
function GV = computeGV(V)

    [VWu, Vbu, VNu] = unpack(V);
    
    GV = mzeros(psize,1);
    
    if hybridmode
        chunkrange = targetchunk; %set outside
    else
        chunkrange = 1:numchunks;
    end

    for chunk = chunkrange
        
        %application of R operator
        rdEdy = cell(numlayers+1,1);
        rdEdx = cell(numlayers, 1);

        GVW = cell(numlayers,1);
        GVb = cell(numlayers,1);
        
        Rx = cell(numlayers,1);
        Ry = cell(numlayers,1);

        yip1 = conv(y{chunk, 1});

        %forward prop:
        Ryip1 = mzeros(layersizes(1), sizechunk);
            
        for i = 1:numlayers

            Ryi = Ryip1;
            Ryip1 = [];

            yi = yip1;
            yip1 = [];

            Rxi = Wu{i}*Ryi + VWu{i}*yi + repmat(Vbu{i}, [1 sizechunk]);
            %Rx{i} = store(Rxi);

            yip1 = conv(y{chunk, i+1});

            if strcmp(layertypes{i}, 'logistic')
                Ryip1 = Rxi.*yip1.*(1-yip1);
            elseif strcmp(layertypes{i}, 'tanh')
                Ryip1 = Rxi.*(1+yip1).*(1-yip1);
            elseif strcmp(layertypes{i}, 'linear')
                Ryip1 = Rxi;
            elseif strcmp( layertypes{i}, 'softmax' )
                Ryip1 = Rxi.*yip1 - yip1.* repmat( sum( Rxi.*yip1, 1 ), [layersizes(i+1) 1] );
            else
                error( 'Unknown/unsupported layer type' );
            end
            
            Rxi = [];

        end
        
        %Backwards pass.  This is where things start to differ from computeHV  Please note that the lower-case r 
        %notation doesn't really make sense so don't bother trying to decode it.  Instead there is a much better
        %way of thinkin about the GV computation, with its own notation, which I talk about in my more recent paper: 
        %"Learning Recurrent Neural Networks with Hessian-Free Optimization"
        for i = numlayers:-1:1

            if i < numlayers
                %logistics:
                if strcmp(layertypes{i}, 'logistic')
                    rdEdx{i} = rdEdy{i+1}.*yip1.*(1-yip1);
                elseif strcmp(layertypes{i}, 'tanh')
                    rdEdx{i} = rdEdy{i+1}.*(1+yip1).*(1-yip1);
                elseif strcmp(layertypes{i}, 'linear')
                    rdEdx{i} = rdEdy{i+1};
                else
                    error( 'Unknown/unsupported layer type' );
                end
            else
                if ~rms
                    %assume canonical link functions:
                    rdEdx{i} = -Ryip1;
                    
                    if strcmp(layertypes{i}, 'linear')
                        rdEdx{i} = 2*rdEdx{i};
                    end
                else
                    RdEdyip1 = -2*Ryip1;
                    
                    if strcmp(layertypes{i}, 'softmax')
                        error( 'RMS error not supported with softmax output' );
                    elseif strcmp(layertypes{i}, 'logistic')
                        rdEdx{i} = RdEdyip1.*yip1.*(1-yip1);
                    elseif strcmp(layertypes{i}, 'tanh')
                        rdEdx{i} = RdEdyip1.*(1+yip1).*(1-yip1);
                    elseif strcmp(layertypes{i}, 'linear')
                        rdEdx{i} = RdEdyip1;
                    else
                        error( 'Unknown/unsupported layer type' );
                    end
                    
                    RdEdyip1 = [];
                    
                end
                
                Ryip1 = [];

            end
            rdEdy{i+1} = [];
            
            rdEdy{i} = Wu{i}'*rdEdx{i};

            yi = conv(y{chunk, i});

            GVW{i} = rdEdx{i}*yi';
            GVb{i} = sum(rdEdx{i},2);

            rdEdx{i} = [];

            yip1 = yi;
            yi = [];
        end
        yip1 = [];
        rdEdy{1} = [];

        GV = GV + pack(GVW, GVb, VNu);
        
    end
    
    GV = GV / conv(numcases);
    
    if hybridmode
        GV = GV * conv(numchunks);
    end
    
    GV = GV - conv(weightcost)*(maskp.*V);

    if autodamp
        GV = GV - conv(lambda)*V;
    end
    
end

    
function [ll, err] = computeLL(params, in, out, nchunks, tchunk)

    ll = 0;
    
    err = 0;
    
    [W,b,N] = unpack(params);
    
    if mod( size(in,2), nchunks ) ~= 0
        error( 'Number of chunks doesn''t divide number of cases!' );
    end    
    
    schunk = size(in,2)/nchunks;
    
    if nargin > 4
        chunkrange = tchunk;
    else
        chunkrange = 1:nchunks;
    end
    
    for chunk = chunkrange
    
        yi = conv(in(:, ((chunk-1)*schunk+1):(chunk*schunk) ));
        outc = conv(out(:, ((chunk-1)*schunk+1):(chunk*schunk) ));

        for i = 1:numlayers
            xi = W{i}*yi + repmat(b{i}, [1 schunk]);

            if strcmp(layertypes{i}, 'logistic')
                yi = 1./(1 + exp(-xi));
            elseif strcmp(layertypes{i}, 'tanh')
                yi = tanh(xi);
            elseif strcmp(layertypes{i}, 'linear')
                yi = xi;
            elseif strcmp(layertypes{i}, 'softmax' )
                tmp = exp(xi);
                yi = tmp./repmat( sum(tmp), [layersizes(i+1) 1] );   
                tmp = [];
            end

        end

        if rms || strcmp( layertypes{numlayers}, 'linear' )
            
            ll = ll + double( -sum(sum((outc - yi).^2)) );
            
        else
            if strcmp( layertypes{numlayers}, 'logistic' )
                
                %outc==0 and outc==1 are included in this formula to avoid
                %the annoying case where you have 0*log(0) = 0*-Inf = NaN
                %ll = ll + double( sum(sum(outc.*log(yi + (outc==0)) + (1-outc).*log(1-yi + (outc==1)))) );
                
                %this version is more stable:
                ll = ll + double(sum(sum(xi.*(outc - (xi >= 0)) - log(1+exp(xi - 2*xi.*(xi>=0))))));
                
                
            elseif strcmp( layertypes{numlayers}, 'softmax' )
                
                ll = ll + double(sum(sum(outc.*log(yi))));
                
            end
        end
        xi = [];

        if strcmp( errtype, 'class' )
            %err = 1 - double(sum( sum(outc.*yi,1) == max(yi,[],1) ) )/size(in,2);
            err = err + double(sum( sum(outc.*yi,1) ~= max(yi,[],1) ) ) / size(in,2);
        elseif strcmp( errtype, 'L2' )
            err = err + double(sum(sum((yi - outc).^2, 1))) / size(in,2);
        elseif strcmp( errtype, 'none')
            %do nothing
        else
            error( 'Unrecognized error type' );
        end
        %err = double(   (mones(1,size(in,1))*((yi - out).^2))*mones(size(in,2),1)/conv(size(in,2))  );
        
        outc = [];
        yi = [];
    end

    ll = ll / size(in,2);
    
    if nargin > 4
        ll = ll*nchunks;
        err = err*nchunks;
    end
    
    ll = ll - 0.5*weightcost*double(params'*(maskp.*params));

end


function yi = computePred(params, in) %for checking G computation using finite differences
    
    [W, b, N] = unpack(params);
    
    yi = in;
        
    for i = 1:numlayers
        xi = W{i}*yi + repmat(b{i}, [1 size(in,2)]);
        
        if i < numlayers
            if strcmp(layertypes{i}, 'logistic')
                yi = 1./(1 + exp(-xi));
            elseif strcmp(layertypes{i}, 'tanh')
                yi = tanh(xi);
            elseif strcmp(layertypes{i}, 'linear')
                yi = xi;
            elseif strcmp(layertypes{i}, 'softmax' )
                tmp = exp(xi);
                yi = tmp./repmat( sum(tmp), [layersizes(i+1) 1] );   
                tmp = [];
            end
        else
            yi = xi;
        end
        
    end
end



maskp = mones(psize,1);
[maskW, maskb, maskN] = unpack(maskp);
disp('not masking out the weight-decay for biases');
for i = 1:length(maskb)
    %maskb{i}(:) = 0; %uncomment this line to apply the l_2 only to the connection weights and not the biases
end
maskp = pack(maskW,maskb,maskN);


indata = single(indata);
outdata = single(outdata);
intest = single(intest);
outtest = single(outtest);


function outputString( s )
    fprintf( 1, '%s\n', s );
%    fprintf( fid, '%s\r\n', s );
end



fid = fopen( [runName '.txt'], 'a' );

outputString( '' );
outputString( '' );
outputString( '==================== New Run ====================' );
outputString( '' );
outputString( ['Start time: ' datestr(now)] );
outputString( '' );
outputString( ['Description: ' runDesc] );
outputString( '' );


ch = mzeros(psize, 1);

if ~isempty( resumeFile )
    outputString( ['Resuming from file: ' resumeFile] );
    outputString( '' );
    
    load( resumeFile );
    
    ch = conv(ch);

    epoch = epoch + 1;
else
    
    lambda = initlambda;
    
    llrecord = zeros(maxepoch,2);
    errrecord = zeros(maxepoch,2);
    lambdarecord = zeros(maxepoch,1);
    timess = zeros(maxepoch,1);
    
    totalpasses = 0;
    epoch = 1;
    
end

if isempty(paramsp)
    if ~isempty(Win)
        paramsp = pack(Win,bin,Nin);
        clear Win bin
    else
        
        %SPARSE INIT:
        paramsp = zeros(psize,1); %not mzeros
        
        [Wtmp,btmp,Ntmp] = unpack(paramsp);
        
        numconn = 15;
        
        for i = 1:numlayers
 
            initcoeff = 1;

            if i > 1 && strcmp( layertypes{i-1}, 'tanh' )
                initcoeff = 0.5*initcoeff;
            end
            if strcmp( layertypes{i}, 'tanh' )
                initcoeff = 0.5*initcoeff;
            end
            
            if strcmp( layertypes{i}, 'tanh' )
                btmp{i}(:) = 0.5;
            end
            
            %outgoing
            %{
            for j = 1:layersizes(i)
                idx = ceil(layersizes(i+1)*rand(1,numconn));
                Wtmp{i}(idx,j) = randn(numconn,1)*coeff;
            end
            %}
            
            %incoming
            for j = 1:layersizes(i+1)
                idx = ceil(layersizes(i)*rand(1,numconn));
                Wtmp{i}(j,idx) = randn(numconn,1)*initcoeff;
            end
			
			%batch normalization
			for j = 1:numlayers+1
				Ntmp{i}(:,1) = ones(layersizes(i),1);
				Ntmp{i}(:,2) = zeros(layersizes(i),1);
			end
            
        end
        
        paramsp = pack(Wtmp, btmp, Ntmp);
        
        clear Wtmp btmp Ntmp
    end
    
elseif size(paramsp,1) ~= psize || size(paramsp,2) ~= 1
    error( 'Badly sized initial parameter vector.' );
else
    paramsp = conv(paramsp);
end

outputString( 'Initial constant values:' );
outputString( '------------------------' );
outputString( '' );
for i = 1:length(rec_constants)
    outputString( [rec_constants{i} ': ' num2str(eval( rec_constants{i} )) ] );
end

outputString( '' );
outputString( '=================================================' );
outputString( '' );

% FOR_BN
% z is normally used as input for a neuron: here, xi is used
% xnorm is used for whitened input
% xf is for gamma*xnorm+beta (f = final)
% activation of a neuron is yip1

% FOR_BN:
% the code runs for one iteration but has a dimension matching error in the second.
% there might be a mistake in the calculation of gradients

for epoch = epoch:maxepoch
    tic

    targetchunk = mod(epoch-1, numchunks)+1;
    
    [Wu, bu, Nu] = unpack(paramsp);

    y = cell(numchunks, numlayers+1);
    x = cell(numchunks, numlayers+1);
    
    if storeD
        dEdy = cell(numchunks, numlayers+1);
        dEdx = cell(numchunks, numlayers);
    end


    grad = mzeros(psize,1);
    grad2 = mzeros(psize,1);
    
    ll = 0;

    %forward prop:
    %index transition takes place at nonlinearity
    for chunk = 1:numchunks
        
        y{chunk, 1} = store(indata(:, ((chunk-1)*sizechunk+1):(chunk*sizechunk) ));
        yip1 = conv( y{chunk, 1} );
		
		% store yip1, ynorms
		yip1s = cell(numlayers,1);
		ynorms = cell(numlayers, 1);

		% normalize inputs
		% batch_norm: Normalization
		gamma1 = Nu{1}(:,1);
		beta1 = Nu{1}(:,2);

% 		mean1 = mean(yip1,2);
% 		std1 = var(yip1,0,2);
% 		std1 = sqrt(std1 + 0.00001);
% 		%ynorm = (yip1-mean1)./std1;	% ynorm is normalized input (x^ in normal terminology)
% 		ynorm = bsxfun(@rdivide,bsxfun(@minus,yip1,mean1),std1);
%         max(ynorm(:))
%         min(ynorm(:))
% 		means{1} = mean1;
% 		stds{1} = std1;

		yip1s{1} = yip1;
		ynorms{1} = yip1;
		

        dEdW = cell(numlayers, 1);
        dEdb = cell(numlayers, 1);
		dEdN = cell(numlayers, 1);

        dEdW2 = cell(numlayers, 1);
        dEdb2 = cell(numlayers, 1);
		dEdN2 = cell(numlayers, 1);
		
		means = cell(numlayers, 1);
		stds = cell(numlayers, 1);

        for i = 1:numlayers
			
            yi = yip1;	% yi is x in normal terminology
			
            yip1 = [];
            xi = Wu{i}*yi + repmat(bu{i}, [1 sizechunk]); % xi is z in normal terminology
			

            if strcmp(layertypes{i}, 'logistic')
                yip1 = 1./(1 + exp(-xi));	% yip1 is _y
            elseif strcmp(layertypes{i}, 'tanh')
                yip1 = tanh(xi);
            elseif strcmp(layertypes{i}, 'linear')
                yip1 = xi;
            elseif strcmp( layertypes{i}, 'softmax' )
                tmp = exp(xi);
                yip1 = tmp./repmat( sum(tmp), [layersizes(i+1) 1] );
                tmp = [];
            else
                error( 'Unknown/unsupported layer type' );
            end
			
			% FOR_BN: how to extract gamma, beta for each layer
			if i ~= numlayers
				gammai = Nu{i+1}(:,1);
				betai = Nu{i+1}(:,2);

				% batch_norm: Normalization
				meani = mean(yip1,2);
% 				stdi = std(yip1,0,2);
% 				stdi = stdi + 1e-6;
                stdi = sqrt(var(yip1')+0.000001)';
%				ynorm = (yip1-meani)./stdi;	% ynorm is normalized input (x^ in normal terminology)
				ynorm = bsxfun(@rdivide,bsxfun(@minus,yip1,meani),stdi);
%                 max(ynorm(:))
%                 min(ynorm(:))
%                ynorm = bsxfun(@rdivide,bsxfun(@minus,yip1,mean(yip1,2)),sqrt(var(yip1')'+0.1));

				means{i+1} = meani;
				stds{i+1} = stdi;

				yip1s{i+1} = yip1;
				ynorms{i+1} = ynorm;
				
%				yf = ynorm.*gammai+betai;	% yf is the final normalized activation
                yf=bsxfun(@plus,bsxfun(@times,ynorm,gammai),betai);
            else 
				yf = yip1;
			end
            
            y{chunk, i+1} = store(yip1);
        end

        %back prop:
        %cross-entropy for logistics:
        %dEdy{numlayers+1} = outdata./y{numlayers+1} - (1-outdata)./(1-y{numlayers+1});
        %cross-entropy for softmax:
        %dEdy{numlayers+1} = outdata./y{numlayers+1};

        if hybridmode && chunk ~= targetchunk
            y{chunk, numlayers+1} = []; %save memory
        end

        outc = conv(outdata(:, ((chunk-1)*sizechunk+1):(chunk*sizechunk) ));
        
        if rms || strcmp( layertypes{numlayers}, 'linear' )
            ll = ll + double( -sum(sum((outc - yip1).^2)) );
        else
            if strcmp( layertypes{numlayers}, 'logistic' )
                %ll = ll + double( sum(sum(outc.*log(yip1 + (outc==0)) + (1-outc).*log(1-yip1 + (outc==1)))) );
                %more stable:
                ll = ll + sum(sum(xi.*(outc - (xi >= 0)) - log(1+exp(xi - 2*xi.*(xi>=0)))));                
            elseif strcmp( layertypes{numlayers}, 'softmax' )
                ll = ll + double(sum(sum(outc.*log(yip1))));
            end
        end
        % xi = [];
        
        for i = numlayers:-1:1

			if i < numlayers
				gammai = Nu{i+1}(:,1);
				betai = Nu{i+1}(:,2);
					
				%dEdynorm = dEdyf.*gammai;	% yf would've been defined previously
				dEdynorm = bsxfun(@times,dEdyf,gammai);
%                 max(dEdynorm(:))
%                 min(dEdynorm(:))
                %dEdstds = dEdynorm.*(yip1s{i+1}-means{i+1}).*((-1/2)*(stds{i+1}).^(-3));
                dEdstds = dEdynorm.*bsxfun(@times,bsxfun(@minus,yip1s{i+1},means{i+1}),(-1/2)*(stds{i+1}).^(-1/2));
                %dEdmeans = dEdynorm./(stds{i+1}); %+ -2*dEdstds .* (yip1-means{i});
				dEdmeans=bsxfun(@times,dEdynorm,-1./stds{i+1})./size(dEdynorm,2);
                %dEdyip1 = dEdynorm./stds{i+1} + (yip1s{i+1}-means{i+1}) .* dEdstds * 2 + dEdmeans;
				dEdyip1 = bsxfun(@rdivide,dEdynorm, stds{i+1})./size(yip1s{i+1},2) + (bsxfun(@minus, yip1s{i+1},means{i+1}) .* dEdstds * 2)./size(yip1s{i+1},2)+ dEdmeans./size(yip1s{i+1},2);
                %logistics:
                if strcmp(layertypes{i}, 'logistic')
                    dEdxi = dEdyip1.*yip1.*(1-yip1);
                elseif strcmp(layertypes{i}, 'tanh')
                    dEdxi = dEdyip1.*(1+yip1).*(1-yip1);
                elseif strcmp(layertypes{i}, 'linear')
                    dEdxi = dEdyip1;
                else
                    error( 'Unknown/unsupported layer type' );
                end
            else
					
                dEdyf = 2*(outc - yf); %mult by 2 because we dont include the 1/2 before
			
                if strcmp( layertypes{i}, 'softmax' )
                    dEdxi = dEdyf.*yip1 - yip1.* repmat( sum( dEdyf.*yip1, 1 ), [layersizes(i+1) 1] );
                    %error( 'RMS error not supported with softmax output' );

                elseif strcmp(layertypes{i}, 'logistic')
                    dEdxi = dEdyf.*yip1.*(1-yip1);
                elseif strcmp(layertypes{i}, 'tanh')
                    dEdxi = dEdyf.*(1+yip1).*(1-yip1);
                elseif strcmp(layertypes{i}, 'linear')
                    dEdxi = dEdyf;
				else
                    error( 'Unknown/unsupported layer type' );
                end

                % dEdyip1 = [];
				% end -- from the previous commented out if branch

                outc = [];
			
			end
            dEdyi = Wu{i}'*dEdxi;

            if storeD && (~hybridmode || chunk == targetchunk)
                dEdx{chunk, i} = store(dEdxf);
                dEdy{chunk, i} = store(dEdyi);
            end

            yi = conv(y{chunk, i});

            if hybridmode && chunk ~= targetchunk
                y{chunk, i} = []; %save memory
            end

            %standard gradient comp:
			dEdW{i} = dEdxi*yi';
            dEdb{i} = sum(dEdxi,2);

            %gradient squared compdjfdjf
            dEdW2{i} = (dEdxi.^2)*(yi.^2)';
            dEdb2{i} = sum(dEdxi.^2,2);

			if i < numlayers
				dEdN{i+1}(:,1) = sum(dEdyf * (ynorms{i+1})',2);
				dEdN{i+1}(:,2) = sum(dEdyf,2);
				dEdN2{i+1}(:,1) = sum((dEdyf * (ynorms{i+1})').^2,2);
				dEdN2{i+1}(:,2) = sum(dEdyf.^2,2);
			end

            dEdyf = dEdyi;

            dEdyi = [];

            yip1 = yi;
            yi = [];
        end
		
		%% FOR BATCH NORMALIZATION: the parameters are updated for the input layer as well
		dEdN{1}(:,1) = sum(dEdyf * (ynorms{1})',2);
		dEdN{1}(:,2) = sum(dEdyf,2);
		dEdN2{1}(:,1) = sum((dEdyf * (ynorms{1})').^2,2);
		dEdN2{1}(:,2) = sum(dEdyf.^2,2);

        if chunk == targetchunk
            gradchunk = pack(dEdW, dEdb, dEdN);
            grad2chunk = pack(dEdW2, dEdb2, dEdN2);
        end


        grad = grad + pack(dEdW, dEdb, dEdN);

        grad2 = grad2 + pack(dEdW2, dEdb2, dEdN2);

        %for checking F:
        %gradouter = gradouter + pack(dEdW, dEdb)*pack(dEdW, dEdb)';

        dEdW = []; dEdb = []; dEdN = []; dEdW2 = []; dEdb2 = []; dEdN2 = [];
    end
    
    grad = grad / conv(numcases);
    grad = grad - conv(weightcost)*(maskp.*paramsp);
    
    grad2 = grad2 / conv(numcases);
    
    gradchunk = gradchunk/conv(sizechunk) - conv(weightcost)*(maskp.*paramsp);
    grad2chunk =   grad2chunk/conv(sizechunk);
    
    ll = ll / numcases;
    
    ll = ll - 0.5*weightcost*double(paramsp'*(maskp.*paramsp));
    
    
    oldll = ll;
    ll = [];
  
   
    
    %the following commented blocks of code are for checking the matrix
    %computation functions using finite differences.  If you ever add stuff
    %to the objective you should check that everything is correct using
    %methods like these (or something similar).  Be warned that if you use
    %hessiancsd (available online) you have to be mindful of what your
    %matrix-vector product implementation does if it's given complex values
    %in the input vector

    %G computation check:
    %{
    estep = 1e-6;
    Gp = zeros(psize);
    dY = zeros(size(outdata,1), psize);
    for n = 1:numcases

        Pbase = computePred( paramsp, conv(indata(:,n)) );

        for j = 1:psize

            Wd = paramsp;
            Wd(j) = Wd(j) + estep;

            dY(:,j) = (computePred( Wd, conv(indata(:,n)) ) - Pbase)/estep;
        end

        %softmax:
        %Gp = Gp + dY'*(diag(y{numlayers+1}(:,n)) - y{numlayers+1}(:,n)*y{numlayers+1}(:,n)')*dY;

        %logistic:
        if ~rms
            %Gp = Gp + dY'*(-diag(  y{numlayers+1}(:,n).*(1-y{numlayers+1}(:,n))  ))*dY;
            Gp = Gp + -2*dY'*dY;
        else
            %{
            yip1 = y{numlayers+1}(:,n);

            dEdyip1 = 2*(outdata(:,n) - yip1); %mult by 2 because we dont include the 1/2 before
            dd = -2*yip1.*(1-yip1);
            dEdxi = dEdyip1.*yip1.*(1-yip1);

            Hm = diag(  dEdyip1.*yip1.*(1-yip1).*(1-2*yip1) + dd.*yip1.*(1-yip1)  );

            dEdyip1 = []; dd = [];
            %}

            yip1 = y{numlayers+1}(:,n);
            Hm = diag( -2* (yip1.*(1-yip1)).^2 );

            %Hm = -2;
            %Gp = Gp + dY'*Hm*dY;
        end

    end
    Gp = Gp / conv(numcases);
    
    lambda = 0.0;
    G = zeros(psize);
    for j = 1:psize

        Wd = zeros(psize,1);
        Wd(j) = 1;

        G(:,j) = computeGV(Wd);
    end
    1==1;    
    %}
    

    %slightly decay the previous change vector before using it as an
    %initialization.  This is something I didn't mention in the paper,
    %and it's not overly important but it can help a lot in some situations 
    %so you should probably use it
    ch = conv(decay)*ch;

    %maxiters is the most important variable that you should try
    %tweaking.  While the ICML paper had maxiters=250 for everything
    %I've since found out that this wasn't optimal.  For example, with
    %pre-trained weights for CURVES, maxiters=150 is better.  And for
    %the FACES dataset you should use something like maxiters=100.
    %Setting it too small or large can be bad to various degrees.
    %Currently I'm trying to automate"this choice, but it's quite hard
    %to come up with a *robust* heuristic for doing this.

    maxiters = 250;
    miniters = 1;
    outputString(['maxiters = ' num2str(maxiters) '; miniters = ' num2str(miniters)]);

    %preconditioning vector.  Feel free to experiment with this.  For
    %some problems (like the RNNs) this style of diaognal precondition
    %doesn't seem to be beneficial.  Probably because the parameters don't
    %exibit any obvious "axis-aligned" scaling issues like they do with
    %standard deep neural nets
    precon = (grad2 + mones(psize,1)*conv(lambda) + maskp*conv(weightcost)).^(3/4);
    %precon = mones(psize,1);

    [chs, iterses] = conjgrad_1( @(V)-computeGV(V), grad, ch, ceil(maxiters), ceil(miniters), precon );

    ch = chs{end};
    iters = iterses(end);

    totalpasses = totalpasses + iters;
    outputString(['CG steps used: ' num2str(iters) ', total is: ' num2str(totalpasses) ]);

    p = ch;
    outputString( ['ch magnitude : ' num2str(double(norm(ch)))] );




    j = length(chs);
    

    %"CG-backtracking":
    %It is not clear what subset of the data you should perform this on.
    %If possible you can use the full training set, as the uncommented block
    %below does.  Otherwise you could use some other set, like the current
    %mini-batch set, although that *could* be worse in some cases.  You can
    %also try not using it at all, or implementing it better so that it doesn't
    %require the extra storage

    %version with no backtracking:
    %{
    [ll, err] = computeLL(paramsp + chs{j}, indata, outdata, numchunks);
    %}
    
    %current mini-batch version:
    %{
    [ll_chunk, err_chunk] = computeLL(paramsp + p, indata, outdata, numchunks, targetchunk);
    [oldll_chunk, olderr_chunk] = computeLL(paramsp, indata, outdata, numchunks, targetchunk);

    for j = (length(chs)-1):-1:1
        [lowll_chunk, lowerr_chunk] = computeLL(paramsp + chs{j}, indata, outdata, numchunks, targetchunk);

        if ll_chunk > lowll_chunk
            j = j+1;
            break;
        end

        ll_chunk = lowll_chunk;
        err_chunk = lowerr_chunk;
    end
    if isempty(j)
        j = 1;
    end
    [ll, err] = computeLL(paramsp + chs{j}, indata, outdata, numchunks);
    %}

    %full training set version:
    [ll, err] = computeLL(paramsp + p, indata, outdata, numchunks);
    for j = (length(chs)-1):-1:1
        [lowll, lowerr] = computeLL(paramsp + chs{j}, indata, outdata, numchunks);

        if ll > lowll
            j = j+1;
            break;
        end

        ll = lowll;
        err = lowerr;
    end
    if isempty(j)
        j = 1;
    end
    
    p = chs{j};
    outputString( ['Chose iters : ' num2str(iterses(j))] );


    [ll_chunk, err_chunk] = computeLL(paramsp + chs{j}, indata, outdata, numchunks, targetchunk);
    [oldll_chunk, olderr_chunk] = computeLL(paramsp, indata, outdata, numchunks, targetchunk);

    %disabling the damping when computing rho is something I'm not 100% sure
    %about.  It probably doesn't make a huge difference either way.  Also this
    %computation could probably be done on a different subset of the training data
    %or even the whole thing
    autodamp = 0;
    denom = -0.5*double(chs{j}'*computeGV(chs{j})) - double(grad'*chs{j});
    autodamp = 1;
    rho = (oldll_chunk - ll_chunk)/denom;
    if oldll_chunk - ll_chunk > 0
        rho = -Inf;
    end

    outputString( ['rho = ' num2str(rho)] );

    chs = [];


    %bog-standard back-tracking line-search implementation:
    rate = 1.0;

    c = 10^(-2);
    j = 0;
    while j < 60

        if ll >= oldll + c*rate*double(grad'*p)
            break;
        else
            rate = 0.8*rate;
            j = j + 1;
            %outputString('#');
        end

        %this is computed on the whole dataset.  If this is not possible you can
        %use another set such the test set or a seperate validation set
        [ll, err] = computeLL(paramsp + conv(rate)*p, indata, outdata, numchunks);
    end

    if j == 60
        %completely reject the step
        j = Inf;
        rate = 0.0;
        ll = oldll;
    end

    outputString( ['Number of reductions : ' num2str(j) ', chosen rate: ' num2str(rate)] );


    %the damping heuristic (also very standard in optimization):
    if autodamp
        if rho < 0.25 || isnan(rho)
            lambda = lambda*boost;
        elseif rho > 0.75
            lambda = lambda*drop;
        end
        outputString(['New lambda: ' num2str(lambda)]);
    end
        

    %Parameter update:
    paramsp = paramsp + conv(rate)*p;

    lambdarecord(epoch,1) = lambda;

    llrecord(epoch,1) = ll;
    errrecord(epoch,1) = err;
    timess(epoch) = toc;
    outputString( ['epoch: ' num2str(epoch) ', Log likelihood: ' num2str(ll) ', error rate: ' num2str(err) ] );

    [ll_test, err_test] = computeLL(paramsp, intest, outtest, numchunks_test);
    llrecord(epoch,2) = ll_test;
    errrecord(epoch,2) = err_test;
    outputString( ['TEST Log likelihood: ' num2str(ll_test) ', error rate: ' num2str(err_test) ] );
    
    outputString( ['Error rate difference (test - train): ' num2str(err_test-err)] );
    
    outputString( '' );

    pause(0)
    drawnow
    
    tmp = paramsp;
    paramsp = single(paramsp);
    tmp2 = ch;
    ch = single(ch);
    save( [runName '_nnet_running.mat'], 'paramsp', 'ch', 'epoch', 'lambda', 'totalpasses', 'llrecord', 'timess', 'errrecord', 'lambdarecord' );
    if mod(epoch,10) == 0
        save( [runName '_nnet_epoch' num2str(epoch) '.mat'], 'paramsp', 'ch', 'epoch', 'lambda', 'totalpasses', 'llrecord', 'timess', 'errrecord', 'lambdarecord' );
    end
    paramsp = tmp;
    ch = tmp2;

    clear tmp tmp2

end

paramsp = double(paramsp);

outputString( ['Total time: ' num2str(sum(timess)) ] );

fclose(fid);

end