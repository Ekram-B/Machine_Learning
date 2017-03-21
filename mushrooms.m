%%
D_char = textscan(fopen('mushrooms.csv','rt'),'%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c');

maps = cell(length(D_char),1);
for i = 1:length(D_char)
    maps{i} = unique(D_char{i});
end

D = zeros(length(D_char{1}),length(D_char));
for i = 1:size(D,1)
    for j = 1:size(D,2)
        D(i,j) = find(maps{j} == D_char{j}(i));
    end
end

%%
D = D(randperm(size(D,1)),:);

D_train = D(1:6499,:);
D_valid = D(6500:end,:);

numVals = max(D,1); % This will be needed when fitting the distributions as it indicates the range of each random variable.

