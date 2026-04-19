%% TD3 SPEED CONTROL - PMSM
% FINAL TUNED VERSION

clear; clc; close all;

%% 1. MODEL AND AGENT BLOCK
mdl = "new24a";
agentBlk = "new24a/RL Agent";

%% 2. OBSERVATION SPACE
% [speed_error, rotor_speed, iq] — all normalized to [-1,1]
obsInfo = rlNumericSpec([3 1], ...
    'LowerLimit', [-1; -1; -1], ...
    'UpperLimit',  [ 1;  1;  1]);
obsInfo.Name = "observations";

%% 3. ACTION SPACE
% iq_ref — keep same
actInfo = rlNumericSpec([1 1], ...
    'LowerLimit', -10, ...
    'UpperLimit',  10);
actInfo.Name = "iq_ref";

%% 4. CREATE ENVIRONMENT
env = rlSimulinkEnv(mdl, agentBlk, obsInfo, actInfo);

%% 5. DIMENSIONS
obsDim = 3;
actDim = 1;

%% 6. ACTOR NETWORK
actorNet = [
    featureInputLayer(obsDim, "Name", "state")
    fullyConnectedLayer(256, "Name", "fc1")
    reluLayer("Name", "relu1")
    fullyConnectedLayer(256, "Name", "fc2")
    reluLayer("Name", "relu2")
    fullyConnectedLayer(actDim, "Name", "fc3")
    tanhLayer("Name", "tanh")
];

actorOpts_net = rlRepresentationOptions(...
    'LearnRate', 1e-4, ...
    'GradientThreshold', 1);

actor = rlDeterministicActorRepresentation(...
    actorNet, obsInfo, actInfo, ...
    'Observation', {'state'}, ...
    'Action', {'tanh'}, ...
    actorOpts_net);

%% 7. CRITIC NETWORK
statePath = [
    featureInputLayer(obsDim, "Name", "state")
    fullyConnectedLayer(256, "Name", "sfc1")
    reluLayer("Name", "srelu1")
    fullyConnectedLayer(256, "Name", "sfc2")
    reluLayer("Name", "srelu2")
];

actionPath = [
    featureInputLayer(actDim, "Name", "action")
    fullyConnectedLayer(256, "Name", "afc1")
    reluLayer("Name", "arelu1")
];

commonPath = [
    additionLayer(2, "Name", "add")
    fullyConnectedLayer(256, "Name", "cfc1")
    reluLayer("Name", "crelu1")
    fullyConnectedLayer(1, "Name", "output")
];

criticNet = layerGraph(statePath);
criticNet = addLayers(criticNet, actionPath);
criticNet = addLayers(criticNet, commonPath);
criticNet = connectLayers(criticNet, "srelu2", "add/in1");
criticNet = connectLayers(criticNet, "arelu1", "add/in2");

criticOpts_net = rlRepresentationOptions(...
    'LearnRate', 3e-4, ...
    'GradientThreshold', 1);

critic = rlQValueRepresentation(criticNet, obsInfo, actInfo, ...
    'Observation', {'state'}, ...
    'Action', {'action'}, ...
    criticOpts_net);

%% 8. TD3 AGENT OPTIONS
agentOpts = rlTD3AgentOptions;

agentOpts.SampleTime                = 1e-3;
agentOpts.ExperienceBufferLength    = 1e6;
agentOpts.MiniBatchSize             = 256;
agentOpts.DiscountFactor            = 0.99;
agentOpts.TargetSmoothFactor        = 1e-3;
agentOpts.PolicyUpdateFrequency     = 2;
agentOpts.NumWarmStartSteps         = 2000;

agentOpts.ExplorationModel.Variance          = 0.1;
agentOpts.ExplorationModel.VarianceDecayRate = 1e-4;
agentOpts.ExplorationModel.VarianceMin       = 0.02;

agentOpts.TargetPolicySmoothModel.Variance          = 0.1;
agentOpts.TargetPolicySmoothModel.VarianceDecayRate = 1e-4;
agentOpts.TargetPolicySmoothModel.VarianceMin       = 0.01;

actorOpts  = rlOptimizerOptions('LearnRate', 1e-4, 'GradientThreshold', 1);
criticOpts = rlOptimizerOptions('LearnRate', 3e-4, 'GradientThreshold', 1);
agentOpts.ActorOptimizerOptions  = actorOpts;
agentOpts.CriticOptimizerOptions = criticOpts;

%% 9. CREATE TD3 AGENT
agent = rlTD3Agent(actor, critic, agentOpts);

%% 10. TRAINING OPTIONS
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',                1500, ...   % more episodes for full profile
    'MaxStepsPerEpisode',         3000, ...   % 3 seconds per episode
    'ScoreAveragingWindowLength', 30, ...
    'Verbose',                    true, ...
    'Plots',                      'training-progress', ...
    'StopTrainingCriteria',       'AverageReward', ...
    'StopTrainingValue',          2500);

%% 11. RESET FUNCTION
env.ResetFcn = @(in) localResetFcn(in);

%% 12. TRAIN
trainingStats = train(agent, env, trainOpts);

%% 13. SAVE
save('td3_pmsm_agent.mat', 'agent', 'trainingStats');
disp('Agent saved successfully.');

%% LOCAL FUNCTIONS
function in = localResetFcn(in)

    % Washing machine speed profiles
    mode = randi(4);   % randomly select operating mode

    switch mode
        case 1
            % Gentle wash
            ref_speed = 300 + 200*rand;    % 300-500 rpm
        case 2
            % Normal wash
            ref_speed = 500 + 300*rand;    % 500-800 rpm
        case 3
            % Fast wash
            ref_speed = 800 + 200*rand;    % 800-1000 rpm
        case 4
            % Spin cycle
            ref_speed = 600 + 400*rand;    % 600-1000 rpm
    end

    step_time = 0.5 + 0.5*rand;    % step at 0.5-1.0s

    fprintf('Mode %d | ref_speed=%.1f RPM | step_time=%.3f s\n', ...
             mode, ref_speed, step_time);

    in = setBlockParameter(in, ...
        "new24a/Step", ...
        "After", num2str(ref_speed));

    in = setBlockParameter(in, ...
        "new24a/Step", ...
        "Time", num2str(step_time));

end
% Test reset function manually
in = Simulink.SimulationInput('new24a');
in = localResetFcn(in);

% Check what values were set
disp(getBlockParameter(in, 'new24a/Step', 'After'));
disp(getBlockParameter(in, 'new24a/Step', 'Time'));