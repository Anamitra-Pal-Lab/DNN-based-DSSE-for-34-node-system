%%%%% Code developed by Behrouz Azimian and Dr. Anamitra Pal %%%%%
%%%%% Tutorial on Deep Neural Network based Distribution State Estimation %%%%%
%%%%% SGSMA May 2024 %%%%%

%% removing all variables and closing all plot windows %%
clear
clc
close all


%% OpenDSS-MATLAB interface %%
DSSObj = actxserver('OpenDSSEngine.DSS');
if ~DSSObj.Start(0)
    disp('Unable to start the OpenDSS Engine')
    return
end
DSSText = DSSObj.Text;
DSSCircuit = DSSObj.ActiveCircuit;
DSSBus = DSSCircuit.ActiveBus;
DSSLines = DSSCircuit.Lines;
DSSSolution = DSSCircuit.Solution;
DSSActiveCktElement = DSSCircuit.ActiveCktElement; 


%% compiling the corresponding OpenDSS model %%
DSSText.Command = 'Compile {C:\Users\bazimian\Desktop\tutorial\Run_IEEE34Mod2.dss}';
%%

%% fixing random see for reproducibility %%
rng default



%% loading the default consumption values of the distribution model %%
Load_interface = DSSCircuit.Loads;
all_load_names = Load_interface.AllNames;
i=1;
Load_interface.First;
while  i <= length(all_load_names)
    all_loads_kw(i,1) = Load_interface.kW;
    all_loads_kvar(i,1) = Load_interface. kvar;
    all_loads_bus_names{i,1} = DSSActiveCktElement.BusNames{1};
    Load_interface.Next;
    i = i+1;
end


%% loading Pecan Street measurements %%
load pecan_texas_minutely_processed



%% choose target hour
choose_hour = ' 15:';


%% mapping Pecan Street consumption readings to the OpenDSS model default values %%
chosen_hour_index = find(contains(dates, choose_hour));
dates_chosen_hour = dates(chosen_hour_index);
number_of_loads_assigned_to_transformer = 5;    % choose based on the number of loads assigned to a distribution transformer
mc_samples = 10000;
for i = 1:length(all_load_names)
    rand_load_index = randi([1,size(use,2)],[1 number_of_loads_assigned_to_transformer]); 
    temp_use = use(chosen_hour_index,rand_load_index);
    counter = 1;
    for j = 1:15:length(dates_chosen_hour)
        quarterly_smart_meter_data_use(counter,:) = mean(temp_use(j:j+14,:));
        counter = counter + 1;
    end
    shifting_factor = all_loads_kw(i,1) - max(sum(quarterly_smart_meter_data_use,2));
    if shifting_factor > 0
        quarterly_smart_meter_data_use_all(:,i) = sum(quarterly_smart_meter_data_use,2)+shifting_factor;
        Kernel_Model = fitdist(sum(quarterly_smart_meter_data_use,2)+shifting_factor,'kernel','Kernel','normal'); 
        Kernel_Model = truncate(Kernel_Model,0.001,1000);  % making sure no negative consumption values are generated from the fitted KDE model
        all_loads_kw_new(:,i) = random(Kernel_Model,[mc_samples,1]);
        all_loads_kvar_new(:,i) = all_loads_kw_new(:,i)*tan(0.3176*rand(1,1));   % assigning random power factor betweeen 0.95 and 1
    else
        scaling_factor = all_loads_kw(i,1)/max(sum(quarterly_smart_meter_data_use,2));
        quarterly_smart_meter_data_use_all(:,i) = sum(quarterly_smart_meter_data_use,2)*scaling_factor;
        Kernel_Model = fitdist(sum(quarterly_smart_meter_data_use,2)*scaling_factor,'kernel','Kernel','normal'); 
        Kernel_Model = truncate(Kernel_Model,0.001,1000); % making sure no negative consumption values are generated from the fitted KDE model
        all_loads_kw_new(:,i) = random(Kernel_Model,[mc_samples,1]);
        all_loads_kvar_new(:,i) = all_loads_kw_new(:,i)*tan(0.3176*rand(1,1));  % assigning random power factor betweeen 0.95 and 1
    end
    random_index_each_load(i,:) = rand_load_index;
end


%% making sure there are no negative values generated for consumption values
temp = find (all_loads_kw_new<0);
if isempty(temp) == 0
    disp ('negative value found in kw')
end
temp = find (all_loads_kvar_new<0);
if isempty(temp) == 0
    disp ('negative value found in kvar')
end

%% cleaning solar generation Pecan Street measurements
[ir, ic] = find(solar<1e-3); % ensuring there are no negative values for solar PV generation
solar(ir,ic) = 0;

 
%% finding the nodes with solar PV generation and the corresponding default values%%
PV_interface = DSSCircuit.Generators;
all_pv_names = PV_interface.AllNames;
PV_interface.First;
for i= 1:size(all_pv_names)
    all_pvs_kw(i,1) = PV_interface.kW;
    temp = strfind(DSSActiveCktElement.BusNames{1},'.');
    if length(temp) > 1
        all_pvs_bus_names{i,1} = DSSActiveCktElement.BusNames{1}(1:temp(1)-1); 
    else
        all_pvs_bus_names{i,1} = DSSActiveCktElement.BusNames{1};
    end
    PV_interface.Next;
end



%% mapping Pecan Street solar PV generation readings to OpenDSS model default values %%
[dummy , dummy, ib] = intersect(all_pvs_bus_names,all_loads_bus_names,'stable');
counter = 1;
for i = 1:length(all_pv_names)
   temp_solar = solar(chosen_hour_index,random_index_each_load(ib(counter),:));
   counter2 = 1;
   for j = 1:15:length(dates_chosen_hour)
        quarterly_smart_meter_data_solar(counter2,:) = mean(temp_solar(j:j+14,:));
        counter2 = counter2 + 1;
    end
    shifting_factor = all_pvs_kw(i,1) - max(sum(quarterly_smart_meter_data_solar,2));
    Kernel_Model = fitdist(sum(quarterly_smart_meter_data_solar,2)+shifting_factor,'kernel','Kernel','normal'); 
    truncate(Kernel_Model,0.001,1000);
    all_pvs_kw_new(:,i) = random(Kernel_Model,[mc_samples,1]);
    counter = counter +1;
end


%% Monte Carlo simulation, solving power flow for time series consumption/generation scenarios  %%
i_notconverged = 0;

for j = 1:size(all_loads_kw_new,1)
    clear DSSOBj DSSCircuit DSSActiveCktElement DSSLines DSSSolution Load_interface
    DSSText.Command = 'Compile {C:\Users\bazimian\Desktop\tutorial\Run_IEEE34Mod2.dss}';
    DSSCircuit = DSSObj.ActiveCircuit;
    DSSActiveCktElement = DSSCircuit.ActiveCktElement;
    DSSLines = DSSCircuit.Lines;
    Load_interface = DSSCircuit.Loads;

    
    % varying feeder head voltage phasor to mimic transmission system variations
    DSSVsource = DSSCircuit.Vsource;
    r = 0.997 + (1.003-0.997)*rand(1);
    DSSVsource.pu =  1.05*r;
    r = 0.965 + (1.035-0.965)*rand(1);
    DSSVsource.AngleDeg = 30*r;
    
    
    % varying the loads according to Pecan street readings
    for k = 1:length(all_load_names)   
        DSSText.command=[[char('load.'), all_load_names{k}, char('.kW=')]  num2str(all_loads_kw_new(j,k)) ' kvar='  num2str(all_loads_kvar_new(j,k)) ''];  % Build bus name and set corresponding kW and kVar
    end
    %varying the pv units according to Pecan street readings
    for k = 1:length(all_pv_names)
       DSSText.command=[[char('generator.'), all_pv_names{k}, char('.kW=')]  num2str(all_pvs_kw_new(j,k))''];  % Build bus name and set corresponding kW and kVar  
    end
    

    % solve power flow
    DSSText.Command = 'Solve';
    
    DSSSolution = DSSCircuit.Solution;
    if ~DSSSolution.Converged                       % Check convergence for power flow calculations
        i_notconverged = i_notconverged + 1;        % calculate the total number of unconverged power flow calculations
    end
    
    % saving all voltages from power flow results
    All_OpenDSS_temp_voltages(:,j) = DSSCircuit.AllBusVolts';

    % saving all line currents from power flow results
    i_Line = DSSLines.First;
    while i_Line > 0
        I_line_bus_names{i_Line,1} = DSSLines.Bus1;
        I_line_bus_names{i_Line,2} = DSSLines.Bus2;
        I_line_bus_names{i_Line,3} = DSSActiveCktElement.DisplayName;
        if length (DSSActiveCktElement.CurrentsMagAng) == 12
            Currents_DSS_Lines(i_Line,:) = DSSActiveCktElement.CurrentsMagAng;       
        elseif length (DSSActiveCktElement.CurrentsMagAng) == 8
            if DSSLines.Bus1(end-2:end) == "1.2" 
                Currents_DSS_Lines(i_Line,1:4) = DSSActiveCktElement.CurrentsMagAng(1:4);
                Currents_DSS_Lines(i_Line,7:10) = DSSActiveCktElement.CurrentsMagAng(5:8);
            elseif DSSLines.Bus1(end-2:end) == "1.3"
                Currents_DSS_Lines(i_Line,1:2) = DSSActiveCktElement.CurrentsMagAng(1:2);
                Currents_DSS_Lines(i_Line,5:6) = DSSActiveCktElement.CurrentsMagAng(3:4);
                Currents_DSS_Lines(i_Line,7:8) = DSSActiveCktElement.CurrentsMagAng(5:6);
                Currents_DSS_Lines(i_Line,11:12) = DSSActiveCktElement.CurrentsMagAng(7:8);
            elseif DSSLines.Bus1(end-2:end) == "2.3"
                Currents_DSS_Lines(i_Line,3:6) = DSSActiveCktElement.CurrentsMagAng(1:4);
                Currents_DSS_Lines(i_Line,9:12) = DSSActiveCktElement.CurrentsMagAng(5:8);
            end
        elseif length (DSSActiveCktElement.CurrentsMagAng) == 4
            if DSSLines.Bus1(end) == '1' 
                Currents_DSS_Lines(i_Line,1:2) = DSSActiveCktElement.CurrentsMagAng(1:2);
                Currents_DSS_Lines(i_Line,7:8) = DSSActiveCktElement.CurrentsMagAng(3:4);
            elseif DSSLines.Bus1(end) == '2'
                Currents_DSS_Lines(i_Line,3:4) = DSSActiveCktElement.CurrentsMagAng(1:2);
                Currents_DSS_Lines(i_Line,9:10) = DSSActiveCktElement.CurrentsMagAng(3:4);
            elseif DSSLines.Bus1(end) == '3'
                Currents_DSS_Lines(i_Line,5:6) = DSSActiveCktElement.CurrentsMagAng(1:2);
                Currents_DSS_Lines(i_Line,11:12) = DSSActiveCktElement.CurrentsMagAng(3:4);
            end
        end
        i_Line = DSSLines.Next;
    end
    vectorized_currents(:,j) = reshape(Currents_DSS_Lines',size(Currents_DSS_Lines,1)*size(Currents_DSS_Lines,2),1);
    Monte_Carlo_iter_number = j                          
end



%% removing the high voltage tranmission line bus
PCC_index_remove = 1:6;
total_buses_removed = [PCC_index_remove];
All_OpenDSS_NodeNames = DSSCircuit.AllNodeNames;
all_node_names = All_OpenDSS_NodeNames(4:end,:);
counter = 1;
for i = 1:size(All_OpenDSS_temp_voltages,1)
    if ~ismember (i,total_buses_removed)
        All_OpenDSS_voltages(counter,:) = All_OpenDSS_temp_voltages(i,:);
        counter = counter +1;
    end
end


%% Converting voltages from rectangular to polar form
counter = 1;
for j = 1:size(all_loads_kw_new,1)
    counter = 1;
    while counter <size(All_OpenDSS_voltages,1)
    temp = All_OpenDSS_voltages(counter,j)+1j*All_OpenDSS_voltages(counter+1,j);
    All_Polar_Voltages(counter,j) = abs(temp);
    All_Polar_Voltages(counter+1,j) = rad2deg(angle(temp));
    counter = counter+2;
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% USER INPUT %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% choose the location (index) of m-PMUs based on "all_node_names" variable for voltage phasor measurements
PMU_index =  [1,2,3,27,28,29,61,62,63]; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
counter = 1;
for i =1:length(PMU_index)
    All_Polar_Voltages_PMU(counter,:) = All_Polar_Voltages(PMU_index(i)*2-1,:);
    All_Polar_Voltages_PMU(counter+1,:) = All_Polar_Voltages(PMU_index(i)*2,:);
    All_Polar_Voltages_PMU_rectangular(counter,:) = All_OpenDSS_voltages(PMU_index(i)*2-1,:);
    All_Polar_Voltages_PMU_rectangular(counter+1,:) = All_OpenDSS_voltages(PMU_index(i)*2,:)';
    counter = counter+2;
end


%% adding TVE error to non-erronous m-PMU voltage phasors
angle_error = 0.025;      % this is actual degree
magnitude_error = 0.05/100;   % this is percentage
TVE_error = 0.05/100;

temp4 = 1;    % to initialize the error and make sure while loop below iterates
for i = 1:size(All_Polar_Voltages_PMU,2)
    counter = 1;
    while counter <= size(All_Polar_Voltages_PMU,1)
        while temp4>TVE_error
        temp1 = normrnd(All_Polar_Voltages_PMU(counter,i),magnitude_error*All_Polar_Voltages_PMU(counter,i)/3);
        temp2 = normrnd(All_Polar_Voltages_PMU(counter+1,i),angle_error/3);
        temp3 =  temp1*exp(1j*temp2*pi/180);
        temp4 = sqrt(((real(temp3)-All_Polar_Voltages_PMU_rectangular(counter,i))^2+(imag(temp3)-All_Polar_Voltages_PMU_rectangular(counter+1,i))^2)/(All_Polar_Voltages_PMU_rectangular(counter,i)^2+All_Polar_Voltages_PMU_rectangular(counter+1,i)^2));
        end
        All_Polar_Voltages_PMU_errornous(counter,i) = temp1;
        All_Polar_Voltages_PMU_errornous(counter+1,i) = temp2;
        counter = counter +2;
        temp4 = 1;
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% USER INPUT %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% choose the location (index) of m-PMUs based on "I_line_bus_names" variable for current phasor measurements
PMU_current_index = [1,10,25];
PMU_current_direction_index = [1 1 1]; % 1 means downstream, 2 means upstream towards the feeder head
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
counter = 1;
for i = 1:length(PMU_current_index)
    if PMU_current_direction_index(i) == 1
        All_Polar_Currents_PMU(counter:counter+5,:) = vectorized_currents(PMU_current_index(i)*12-11:PMU_current_index(i)*12-11+5,:);
        counter = counter +6;
    else
        All_Polar_Currents_PMU(counter:counter+5,:) = vectorized_currents(PMU_current_index(i)*12-5:PMU_current_index(i)*12,:);
        counter = counter +6;
    end
end


%% adding TVE error to non-erronous m-PMU current phasors

counter = 1;
for i =1:size(All_Polar_Currents_PMU,1)/2
    temp1  = All_Polar_Currents_PMU(2*i-1,:).*exp(1j*All_Polar_Currents_PMU(2*i,:)*pi/180);
    All_Polar_Currents_PMU_rectangular(counter,:) = real(temp1);
    All_Polar_Currents_PMU_rectangular(counter+1,:) = imag(temp1);
    counter = counter+2;
end

temp4 = 1;    % to initialize the error and make sure while loop below iterates
targeted_current = All_Polar_Currents_PMU ; 
targeted_current_rectangular = All_Polar_Currents_PMU_rectangular; 
for i = 1:size(All_Polar_Currents_PMU,2)
    counter = 1;
    while counter <= size(All_Polar_Currents_PMU,1)
        while temp4>TVE_error
        temp1 = normrnd(All_Polar_Currents_PMU(counter,i),magnitude_error*All_Polar_Currents_PMU(counter,i)/3);
        temp2 = normrnd(All_Polar_Currents_PMU(counter+1,i),angle_error/3);
        temp3 =  temp1*exp(1j*temp2*pi/180);
        temp4 = sqrt(((real(temp3)-All_Polar_Currents_PMU_rectangular(counter,i))^2+(imag(temp3)-All_Polar_Currents_PMU_rectangular(counter+1,i))^2)/(All_Polar_Currents_PMU_rectangular(counter,i)^2+All_Polar_Currents_PMU_rectangular(counter+1,i)^2));
        end
        All_Polar_Currents_PMU_errornous(counter,i) = temp1;
        All_Polar_Currents_PMU_errornous(counter+1,i) = temp2;
        counter = counter +2;
        temp4 = 1;
    end

end   


%% determining the test percentage of the generated data for testing DNN performance
test_percentage = 20/100; 


%% removing mid nodes and regulator nodes from the output of neural network
counter = 1;
for i = 1:size(all_node_names,1)
    if all_node_names{i,1}(1) == 'm' || all_node_names{i,1}(4) == 'r' 
        mid_node_remove_for_SE(counter) = i;
        counter = counter +1;
    end
end
mid_node_remove_for_SE = unique(mid_node_remove_for_SE);
temp1 = [];
temp1 = [mid_node_remove_for_SE'*2,mid_node_remove_for_SE'*2-1]';
All_Polar_Voltages_removed_mid = All_Polar_Voltages; 
All_Polar_Voltages_removed_mid(temp1,:) = [];   % comment this line if you want to estimate mid bus voltages too
bus_names_temp = all_node_names;
bus_names_temp(mid_node_remove_for_SE) = []; 



%% finding the index of each phase
counter = 1;
phase_A_index = [];
phase_B_index = [];
phase_C_index = [];
for i = 1:size(bus_names_temp,1)
    if bus_names_temp{i}(end) == '1'
        phase_A_index = [phase_A_index,i];
    elseif bus_names_temp{i}(end) == '2'
        phase_B_index = [phase_B_index,i];
    elseif bus_names_temp{i}(end) == '3'
        phase_C_index = [phase_C_index,i];
    end
end
% or manually enter the indexes according to IEEE 34 node system
phase_A_index = [1,4,7,10,14,17,20,23,26,30,31,27,33,36,39,70,42,73,84,45,76,48,51,60,63,66,77,54,57,80];
phase_B_index = [2,5,8,11,13,15,18,21,24,28,32,34,37,40,69,71,43,74,85,46,49,52,61,64,67,78,55,58,81,83];
phase_C_index = [3,6,9,12,16,19,22,25,29,35,38,41,72,44,75,86,47,50,53,62,65,68,79,56,59,82];
phase_A_names = bus_names_temp(phase_A_index);
phase_B_names = bus_names_temp(phase_B_index);
phase_C_names = bus_names_temp(phase_C_index);


%% creating train and test datasets for voltage measurements for DNN training


train_All_Polar_Voltages_PMU_errornous = All_Polar_Voltages_PMU_errornous(:,1:(1-test_percentage)*size(All_Polar_Voltages_PMU_errornous,2));
train_All_Polar_Voltages = All_Polar_Voltages_removed_mid(:,1:(1-test_percentage)*size(All_Polar_Voltages_removed_mid,2));
test_All_Polar_Voltages_PMU_errornous = All_Polar_Voltages_PMU_errornous(:,(1-test_percentage)*size(All_Polar_Voltages_PMU_errornous,2)+1:end);
test_All_Polar_Voltages = All_Polar_Voltages_removed_mid(:,(1-test_percentage)*size(All_Polar_Voltages_removed_mid,2)+1:end);



%% converting voltage phasors to p.u. and radians for better performance for DNN training
Vbase1 = 24900/sqrt(3);
Vbase2 = 4160/sqrt(3);
odd_counter = 1:2:size(train_All_Polar_Voltages,1);
even_counter = 2:2:size(train_All_Polar_Voltages,1);
counter = 1;
for i = odd_counter
    if train_All_Polar_Voltages(i,1) > Vbase1*0.7 && train_All_Polar_Voltages(i,1) < Vbase1*1.7
        Vbase_all(counter,1) = Vbase1;
        counter = counter+1;
    elseif train_All_Polar_Voltages(i,1) > Vbase2*0.7 && train_All_Polar_Voltages(i,1) < Vbase2*1.7
        Vbase_all(counter,1) = Vbase2;
        counter = counter+1;
    end
    
end
train_All_Polar_Voltages(odd_counter,:) = train_All_Polar_Voltages(odd_counter,:)./Vbase_all;
train_All_Polar_Voltages(even_counter,:) = train_All_Polar_Voltages(even_counter,:)*pi/180;
test_All_Polar_Voltages(odd_counter,:) = test_All_Polar_Voltages(odd_counter,:)./Vbase_all;
test_All_Polar_Voltages(even_counter,:) = test_All_Polar_Voltages(even_counter,:)*pi/180;
%% creating train and test datasets for current measurements for DNN training

train_All_Polar_Currents_PMU_errornous = All_Polar_Currents_PMU_errornous(:,1:(1-test_percentage)*size(All_Polar_Currents_PMU_errornous,2));
test_All_Polar_Currents_PMU_errornous = All_Polar_Currents_PMU_errornous(:,(1-test_percentage)*size(All_Polar_Currents_PMU_errornous,2)+1:end);


%% concatenating m-PMU voltage and current phasor measurements 
train_voltage_current_PMU_errornous =[train_All_Polar_Voltages_PMU_errornous;train_All_Polar_Currents_PMU_errornous];
test_voltage_current_PMU_errornous = [test_All_Polar_Voltages_PMU_errornous;test_All_Polar_Currents_PMU_errornous];

%% normalization of the input dataset for DNN training
max_input_train = max(train_voltage_current_PMU_errornous,[],2);
min_input_train = min(train_voltage_current_PMU_errornous,[],2);
normalized_train_voltage_current_PMU_errornous = (train_voltage_current_PMU_errornous - min_input_train)./(max_input_train - min_input_train);


 %% exporting the training data for DNN training in .csv format
writematrix([[0:size(normalized_train_voltage_current_PMU_errornous,1)-1];[normalized_train_voltage_current_PMU_errornous']],'train_input.csv')
writematrix([[0:size(train_All_Polar_Voltages,1)-1];[train_All_Polar_Voltages']],'train_output.csv')



%% normalization of input test dataset for DNN testing
normalized_test_voltage_current_PMU_errornous = (test_voltage_current_PMU_errornous - min_input_train)./(max_input_train - min_input_train);
%% exporting the testing data for DNN testing in .csv format
writematrix([[0:size(normalized_test_voltage_current_PMU_errornous,1)-1];[normalized_test_voltage_current_PMU_errornous']],'test_input.csv')
writematrix([[0:size(test_All_Polar_Voltages,1)-1];[test_All_Polar_Voltages']],'test_output.csv')

%% exporting node indexes for each phase for error calculation
writematrix([[0:size(phase_A_index,1)-1];[phase_A_index']],'phase_A_output_indexes.csv')
writematrix([[0:size(phase_B_index,1)-1];[phase_B_index']],'phase_B_output_indexes.csv')
writematrix([[0:size(phase_C_index,1)-1];[phase_C_index']],'phase_C_output_indexes.csv')













%% calculating correlation coefficients

heatmap(corrcoef(All_Polar_Voltages_removed_mid(phase_A_index*2-1,:)'));
% figure
% heatmap(corrcoef(All_Polar_Voltages_removed_mid(phase_A_index*2-1,mc_samples+1:end)'))

% voltage_profile_A_index = [1,4,7,10,14,17,20,23,27,33,36,39,70,42,45,48,51,60,63,66]*2-1; % based on bus_names_temp
% 
% for i=1:size(All_Polar_Voltages_removed_mid,2)
%     plot(All_Polar_Voltages_removed_mid(voltage_profile_A_index,i)/Vbase1)
%     hold on
% end




