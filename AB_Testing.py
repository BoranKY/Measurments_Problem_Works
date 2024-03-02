import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp,shapiro,levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
pd.set_option("display.max_rows",None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%.5f" % x)


df_control = pd.read_excel("datasets/ab_testing.xlsx",sheet_name="Control Group")
df_test = pd.read_excel("datasets/ab_testing.xlsx",sheet_name="Test Group")

df_control.describe().T   # --> Maximum Bidding
df_test.describe().T      # --> Average Bidding

df_test["group"] = "test"
df_control["group"] = "control"

df = pd.concat([df_test,df_control], axis=0, ignore_index=True)
df.head()

#################  Defining the Hypothesis of A/B Testing #################

#### Hypothesis Description ####

# H0: M1 = M2   --> There is no difference between maximum binding and average bidding
# H1: M1 != M2  --> There is a difference between maximum binding and average bidding.

# Analyze the purchase (earnings) averages for the control and test groups.

df.groupby("group").agg({"Purchase":"mean"})



################# Performing Hypothesis Testing  #################
#  Perform assumption checks before hypothesis testing.

# Normal Variance Control TEST GROUP--> shapiro
test_stat, pvalue = shapiro(df.loc[df["group"]=="test","Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))  # Normality assumption is met for the test group
# Normal Variance Control CONTROL GROUP--> shapiro
test_stat,pvalue = shapiro(df.loc[df["group"]=="control","Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))  # Normality assumption is met for the control group


# Variance Homogeneity Control --> levene
test_stat,pvalue = levene(df.loc[df["group"]=="test","Purchase"],
                          df.loc[df["group"]=="control","Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))  # Homogeneity assumption is ensured for the test and control groups


### Since both Normal and Homogeneous variance tests are provided, the ttest method will be applied. ###

test_stat,pvalue = ttest_ind(df.loc[df["group"]=="test","Purchase"],
                          df.loc[df["group"]=="control","Purchase"],
                             equal_var =True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))  # Since the P Value is 0.3493,
# we say that the H0 hypothesis cannot be rejected and there is no difference between them.


################# Conclusion #################

# Since both methods give identical results, the company can choose either method.











