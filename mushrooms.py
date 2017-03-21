# Module declarations
import numpy as np
import csv
from collections import OrderedDict
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------
filename = 'mushrooms.csv'
with open(filename, 'rb') as raw_file:
    raw_data = csv.reader(raw_file, delimiter=',', quoting=csv.QUOTE_NONE)
    data_list = list(raw_data)

ndims = len(data_list[0])
npts = len(data_list)

char_maps = [OrderedDict() for i in range(ndims)] # Mapping of character to integer
reverse_maps = [[] for i in range(ndims)] # Array of all the possible values for each field
data_mat = np.empty((npts,ndims), dtype=np.int32) # Converted matrix of csv
for i,cdata in enumerate(data_list):
    for j,cstr in enumerate(cdata):
        if cstr not in char_maps[j]:
            char_maps[j][cstr] = len(char_maps[j])
            reverse_maps[j].append(cstr)
        data_mat[i,j] = char_maps[j][cstr]
del data_list

np.random.seed(0)
data_perm = np.random.permutation(npts)
data_train = data_mat[data_perm[0:(8*npts/10)],:]
data_test = data_mat[data_perm[(8*npts/10):],:]
data_ranges = data_mat[:,1:].max(axis=0)
#---------------------------------------------------------------------------

# Global vars
features = ["cap-shape", # Feature 1
            "cap-surface", # Feature 2
            "cap-color", # Feature 3 
            "bruises?", # Feature 4 
            "odor", # Feature 5 
            "gill-attachment", # Feature 6 
             "gill-spacing", # Feature 7 
             "gill-size", # Feature 8
             "gill-color", # Feature 9
            "stalk-shape", # Feature 10 
            "stalk-root", # Feature 11 
            "stalk-surface-above-ring", # Feature 12 
            "stalk-surface-below-ring", # Feature 13
            "stalk-color-above-ring", # Feature 14
             "stalk-color-below-ring", # Feature 15
            "veil-type", # Feature 16
            "veil-color", # Feature 17
            "ring-number",# Feature 18 
            "ring-type", # Feature 19
            "spore-print-color", # Feature 20
             "population", # Feature 21 
             "habitat"] # Feature 22

# Functions Defintions
# Partition the training data to include only the poisoned data
# Remark_1 - only need the first feature to determine this
# Remark_2 - using referenced defined in given code
def generate_not_edible_array(data_train, data_mat):
    poison_array = np.where(data_train[:, 0] == 0)[0]
    len_not_edible = len(poison_array)
    poison_mat = np.empty((len_not_edible,len(data_mat[0])), dtype=np.int32)
    j = 0
    for i in poison_array:
        poison_mat[j] = data_train[i]
        j = j + 1
    return poison_mat
# Parition the training data to include only the non edible data
def generate_edible_array(data_train, data_mat):
    edible_array = np.where(data_train[:, 0] == 1)[0]
    len_ed = len(edible_array)
    edible_mat = np.empty((len_ed, len(data_mat[0])), dtype=np.int32)
    j = 0
    for i in edible_array:
        edible_mat[j] = data_train[i]
        j = j + 1  
    return edible_mat

def histogram_gen():
    for i in range(1, len(features) + 1):
        bins = len(char_maps[i])
        ed_hist = np.zeros(bins)
        po_hist = np.zeros(bins)
        for j in range(num_bins):
            ed_hist[j] = len(np.where(edible_mat[:, i] == j)[0])
            po_hist[j] = len(np.where(poison_mat[:, i] == j)[0])
        # Format the histrograms for edible and non edible occurrenced of the feature
        plt.subplot(211)
        plt.ylabel("Number of Occurences")
        plt.ylim((100, 500))
        plt.title("Feature {}: {}. poisonous".format(i, features[i-1]))
        plt.bar(range(num_bins), po_hist)

        plt.subplot(212)
        plt.ylabel("Number of Occurences")
        plt.title("Feature {}: {}. edible".format(i, features[i-1]))
        plt.bar(range(num_bins), ed_hist)
        plt.show()

# Used in baysian analysis
def calculate_priors(a):
    # local variable declarations
    poisonous_features = [[] for i in range(ndims - 1)]  
    edible_features = [[] for i in range(ndims - 1)]  
    pNum = len(poison_partition)  # Poisoned_num
    eNum = len(edible_partition)  # Edible_num
    # routine part
    for i in range(1, len(features) + 1):
        bins = len(char_maps[i])
        poisoned_theta = np.zeros(bins)
        edible_theta = np.zeros(bins)
        for j in range(bins):
            ed_amount = len(np.where(edible_partition[:, i] == j)[0])
            po_amount = len(np.where(poison_partition[:, i] == j)[0])
            edible_theta[j] = (float(ed_amount) + a - 1)/(eNum + bins * a - bins)
            poisoned_theta[j] = (float(po_amount) + a - 1)/(pNum + bins * a - bins)
        poisonous_features[i - 1] = poisoned_theta
        edible_features[i - 1] = edible_theta
    return poisonous_features, edible_features

# Log space translation of priors to ensure precision
def log_sum_exp(poisonous_features, edible_features, data):
    # local variable declarations
    correct = 0
    # Routine part
    for i in range(len(data)):
        lp1 = 0
        lp0 = 0
        for j in range(1, len(data[0])):
            feature_val = data[i][j]
            if edible_features[j - 1][feature_val] != 0:
                lp1 = lp1 + np.log(edible_features[j - 1][feature_val])
            if poisonous_features[j-1][feature_val] != 0:
                lp0 = lp1 + np.log(poisonous_features[j-1][feature_val])
        lp1 = lp1 + np.log(edible_percentage)
        lp0 = lp0 + np.log(poison_percentage)
        
        B = max(lp1, lp0)

        posterior = lp1 - (np.log(np.exp(lp1 - B) + np.exp(lp0 - B)) + B)
        # If determined prosterior is greater than 50 percent, we can perform a negative predication
        if posterior > np.log(0.5):
            if data[i][0] == 1:
                correct = correct + 1
        # If determined prosterior is less than 50 precent, we can perform a positive predication
        if posterior <= np.log(0.5):
            if data[i][0] == 0:
                correct = correct + 1
    return correct
# Use generate probability vs alpha functions
def best_alpha(alpha, data):
    best = -1000000
    best_a = -1000000
    all_alpha = np.zeros(len(alpha))
    size_data = len(data)
    i = 0
    for a in alpha:
        poisonous_features, edible_features = calculate_priors(a)
        amount_accuracy = float(log_sum_exp(poisonous_features, edible_features, data))/size_data
        all_alpha[i] = amount_accuracy
        i += 1
        if amount_accuracy > best:
            best = amount_accuracy
            best_a = a
    return best_a, all_alpha

def biggest_impact(a, poisonous_features, edible_features):
    computed_matrix = []            # Stores lists with structure [i, f_i, abs_value, raw_value]

    for i in range(len(poisonous_features)):
        for j in range(len(poisonous_features[i])):

            edible = 0
            poison = 0

            if np.log(edible_features[i][j]) != 0:
                edible = np.log(edible_features[i][j])

            if np.log(poisonous_features[i][j]) != 0:
                poison = np.log(poisonous_features[i][j])

            value = edible - poison
            abs_value = abs(value)
            computed_matrix.append([i, j, abs_value, value])

    

    abs_sorted = sorted(computed_matrix, key=sort_by_abs)
    raw_sorted = sorted(computed_matrix, key=sort_by_val)

    return abs_sorted, raw_sorted

def sort_by_abs(item):
        return item[2]
def sort_by_val(item):
        return item[3]
# Step 1
# Using references from Prof. code
# determine percentage of data that is edible and not edible - for programming report
# Print out historgrams
poison_partition = generate_not_edible_array(data_train, data_mat)   
edible_partition = generate_edible_array(data_train, data_mat)    
train_num = len(data_train)                       
poison_percentage = float(len(poison_partition))/train_num            
edible_percentage = float(len(edible_partition))/train_num             
print "Out of {}, {} are poisonous and {} are edible".format(train_num, poison_percentage*100, edible_percentage*100)
# Used to compute histograms for programming partition of report
#histogram_gen()

# STEP 2
alpha = np.concatenate((np.arange(1, 2, 0.1), np.arange(2, 1001, 1)))

best_a_train, all_alpha_train = best_alpha(alpha, data_train)
best_a_test, all_alpha_test = best_alpha(alpha, data_test)


plt.xlabel("Alpha values")
plt.ylabel("Probability")
plt.ylim((0.84, 1.00))
plt.plot(alpha, all_alpha_train, label="Train Set")
plt.plot(alpha, all_alpha_test, label="Validation Set")
plt.legend()
plt.show()
# STEP 3







