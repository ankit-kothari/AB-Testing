# How to plan an AB Test?

This is a theoretical article covering the different aspects of AB testing, for a complete working example, please refer to the following article.

[Analyze an A/B test from the popular mobile puzzle game Cookie Cats]. 

https://colab.research.google.com/drive/1hvqRno8u73yeU8pW_gDZ_cD5SMXU5u0v?usp=sharing

## Experiment Design

### What is an A/B/C../N testing experiment?

In an A/B/C../N testing experiment, we are looking to see whether one or more **explanatory variables**, e.g., change in the color of the button, change in the fonts on the webpage, affect the **response variable**. That is the metrics we selected to measure like conversion rate, retention rate, or people buying more stuff.

The people are assigned into at least two different groups, using proper **random sampling** techniques like cluster sampling, stratified sampling depending upon the use case. That way, the groups aren’t biased.

One group acts as the **control group**, which is the group that does nothing, receives nothing, or isn’t changed the way they are working. The other group is called the **treatment group** (also called the experimental group), a group that does something, receives something, or gets a new feature that we are trying to launch. The classic example of this is in medical studies, where the treatment group receives some new drug, and the control group receives a placebo, or sugar pill.

Other methods can be used to make the experiment more reliable. For example, the trial could be **blind or double-blind**.

- A **blind experiment** is when the participants don’t know whether they’re in the control group or the treatment group.
- A **double-blind experiment** is when neither the participants nor the people administering the examination know if the participant belongs to a control or a treatment group.
- **Blocking** The separation of participants into related groups is called blocking. For example, we can block on gender by randomly selecting an equal number of men and women, instead of a truly random sample in which the number of men and women isn’t controlled. For example, If they then treat half of the men and half of the women with the drug, and give a placebo to the other half of the men and the other half of the women, the blocking on gender helps them to see if the drug affects men and women differently.
- **Matched pairs** A matched pairs experiment is a more specific kind of blocking. The participants in the **treatment group and the control group** are matched based on similar characteristics. For example, to see how gender and age change the effect of the blood pressure drug. We could match the ages and genders in the control group with the ages and genders in the **treatment group.** For example, they could put one 18-year-old man in the treatment group and put her matched pair (another 18-year-old man) in the control group. A matched pairs experiment design is an improvement over a completely randomized design. Participants are still randomly assigned to the treatment and control groups, but potentially confounding variables, like age and gender, are controlled and accounted for in the experiment.

### Null Hypothesis and Alternate hypothesis: **What question do we want to answer?**

- **H0 (null hypothesis)**: status quo, we don't expect any change.  The null hypothesis usually states that there is **no difference** between treatment and control groups. (To put this another way, we’re saying our treatment outcome will be statistically similar to our control outcome )
- **HA (alternative hypothesis):** The alternative hypothesis states that **there is a difference** between treatment and control groups. (In other words, the treatment outcome will be statistically different to the control outcome)

  
    <img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/FB43C87D-FC14-4F57-B2EB-149DEF1B5233.jpeg" width="40%">

    Types of Hypotheis Test  

**Few examples,**

**Case 1 (two-tailed test)**

- **H0:**  **Data Science pays as much as software engineer**  ;
- **HA:** **Data Science does not pay as much as a software engineer**.
- This will be two-tailed meaning the data science might be getting more or less than software engineers. The **two-tailed test** is less stringent than the 1-tailed test

**Case 2 (upper-tail test)**

- **H0:**  **Data Science pays less than equal to software engineer**  ;
- **HA:** **Data Science  pays greater software engineer**.
- This is called **upper-tail test(one-tail test)**, we are checking only if data Science pays higher

**Case 3  (lower-tail test)**

- **H0:**  **Data Science pays greater than equal to software engineer**  ;
- **HA:** **Data Science  pays less software engineer**.
- This is called the **lower-tail test(one-tail test)**, we are checking only if data Science pays lower.
- **H0 (null hypothesis)**: status quo, we don't expect any change.  The null hypothesis usually states that there is **no difference** between treatment and control groups. (To put this another way, we’re saying our treatment outcome will be statistically similar to our control outcome )
- **HA (alternative hypothesis):** The alternative hypothesis states that **there is a difference** between treatment and control groups. (In other words, the treatment outcome will be statistically different to the control outcome)

**Few examples,**

**Case 1 (two-tailed test)**

- **H0:**  **Data Science pays as much as software engineer**  ;
- **HA:** **Data Science does not pay as much as a software engineer**.
- This will be two-tailed meaning the data science might be getting more or less than software engineers. The **two-tailed test** is less stringent than the 1-tailed test

**Case 2 (upper-tail test)**

- **H0:**  **Data Science pays less than equal to software engineer**  ;
- **HA:** **Data Science  pays greater software engineer**.
- This is called **upper-tail test(one-tail test)**, we are checking only if data Science pays higher

**Case 3  (lower-tail test)**

- **H0:**  **Data Science pays greater than equal to software engineer**  ;
- **HA:** **Data Science  pays less software engineer**.
- This is called the **lower-tail test(one-tail test)**, we are checking only if data Science pays lower.

**Hoes it look like with python and scipy?**

```python
**#How does it work in python and scipy**
from scipy.stats import mannwhitneyu
stat, p_value = mannwhitneyu(a_dist, b_dist, alternative="greater")  #upper tail test
stat, p_value = mannwhitneyu(a_dist, b_dist, alternative="two-sided")  #two tail test
stat, p_value = mannwhitneyu(a_dist, b_dist, alternative="less")     #lower tail test
```

### **Significance Level** What is the acceptable risk of accepting the alternative hypothesis?

- **Significance (𝛂)** = the probability of rejecting `H0` when `H0` is true, common values are `0.1, 0.05 and 0.1`, depending on the risk of making such mistake, the higher the risk, the lower your `𝛂` should be. These correspond to **confidence intervals** of `90%, 95%, and 99%` respectively.
- **Power (1-𝛃)**
    - The probability of **correctly rejecting  H0 when H0 is false**. It can also be interpreted as the likelihood of observing the effect when there is said effect to be observed. eg: the new recipe with choclate  is, in fact, more popular than the original recipe with vanila.
    - The  common values of **(1-𝛃)**  are `0.8, 0.85, 0.9 and 0.95, 0.99`. This depends on how costly it is to miss this change when there is an actual change, the higher the cost, the lower your `𝜷` should be and hence higher target **statistical power**.
    - A low power will mean our results are less reliable and we may be making false conclusions about our alternative hypothesis. Ideally we want power to be above `0.75–0.8` for a good probability of detecting an effect accurately.
    - Power is analogous to significance, This will enable us to determine sample size needed so we have enough statistical power and significance.

Higher the power better off we are in accepting the alternative hypothesis 

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/4F78D489-6828-4925-A1FF-26FFD92289AF.jpeg" width="40%">

### Metric Selection: How to evaluate the experiment?

**Invariate Metrics**

- Invariate metrics are used for "sanity checks", that is, to make sure our experiment (the way we presented a change to a part of the population **treatment group**, as well as the way we collected the data) is not inherently wrong. Basically, this means we pick metrics which we consider not to change (not to be affected) because of our experiment and later make sure these **metrics don't change drastically between our control and treatment  groups.** This ensures the samples in the control and treatment groups are randomly and evenly distributed and not biased or different.

**Evaluation Metrics What is your success metric?**

- Decide on the metrics which will decide the possible outcome of the ab test
    - Only two possible outcomes, **discrete** alternative, (Yes/No; Click/No-Click)
    - It can be a **continuous** category which covers session time, savings, loss after and before implementing a feature.
- Usually, the categorical feature requires more samples than the continuous features

### Effect Size and Baseline Metric: What is the amount of effect desired?

### **Baseline Metric**

- Before we start our experiment we should know the current value of the metric we're using to evaluate the efficacy of the test i.e.  how these metrics behave before the change - that is, that is called the  **baseline values.**

    **Example:** 

    - So,  if today `30%` of the people that try  vanila cupcakes want to have a second one, the `baseline metric is 0.3`.  With this information, we can determine the scope for  improvement with new experimentation/ideas is `high 70%`.
    - On the other hand, For instance,  if the  baseline metric was instead `95%` like 95 out of 100 people ask for 2nd vanila cupcake then the  recipe is already a success and here is very little room for improvement `5%`, which may or may not justify the effort of running more tests to fine tune the recipe. As a result we can't  expect  improvement greater than 5%. This is agood check to understand the value we can get out of the test and if the level of effort to make a change is worth it or not.

### **Effect Size**

- Effect size is  the magnitude of **difference** between averages/proportions  of **treatment  and control group**. It is the variance in averages between test and control groups divided by the standard deviation of the control i.e. The standardized difference between 2 groups. It has the units stdev. So an effect size of 1 is equal to a difference of 1 stdev. between groups. **In a nutshell “effect size” which is a simple way to measure the effect of a treatment.**

    Example: 

    **Effect size for means**

    - The average of purchases (purchase_mean_control_group) is `0.7` and the standard deviation (purchase_std_control_group) is `0.84`.
    - If we want to increase the purchase_mean to `0.75` in this experiment. We can calculate the effect size like below:

        ```
        **effect_size** = (0.75 - purchase_mean_control_group)/purchase_std_control_group 
        ```

    **Effect size for proportions** 

    ```
    **effect_size** = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2))) 

    **#python implementation**
    import statsmodels.stats.api as sms
    baseline_cvr=0.1
    mini_diff=0.1*baseline_cvr #10% difference we want to see 
    **effect_size**=sms.proportion_effectsize(baseline_cvr, baseline_cvr+mini_diff)
    **effect size for a 10% increase from baseline -0.0326**
    ```

        

### Samples Needed to perform the experiment **How many samples need to be in the experiment?**

- **Effect Size**.  [Effect size](https://machinelearningmastery.com/effect-size-measures-in-python/) is calculated using a specific statistical measure, such as Pearson’s correlation coefficient for the relationship between variables or Cohen’s d for the difference between groups.
- **Sample Size**. The number of observations in the sample. **(n)**
- **Significance**. The significance level used in the statistical test, e.g. alpha. 1% and 5%. These correspond to confidence intervals of 90% and 95%, respectively. **(alpha)**
- **Statistical Power**. The probability of accepting the alternative hypothesis if it is true. **(1-beta)**

There can be a variety of formulas based on different use-case. This **[link](https://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Power/BS704_Power_print.html)** covers almost all the use-cases with examples. The different versions using **statsmodels.stats.api** with  [**Python Implementation**](http://jpktd.blogspot.com/2013/03/statistical-power-in-statsmodels.html) 

Few examples. They will have to be calculated per metric. 

- **This type is applicable for all the categorical and binary metrics (Proportions)**

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-16_at_12.20.38_PM.png" width="40%">
    
    **omega** here is the  minimum detectable change to make this experiment worth it. 

    **omega** is the minimum detectable change

**Sample Code for  calculating minimum samples to compare for two Proportions (Normal Approximation)**

```sql
import statsmodels.stats.api as sms
baseline_cvr=0.1
alpha=0.05
power=0.8
mini_diff=0.1*baseline_cvr
effect_size=sms.proportion_effectsize(baseline_cvr, baseline_cvr+mini_diff)
sample_size=sms.NormalIndPower().solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=1)
print('Required sample size ~ {0:.1f}'.format(sample_size) + ' per group')
#Output:
#Required sample size ~ 14744.1 per group
```

**Sample Code for  calculating minimum samples to compare for two Means (Normal Approximation)**

- **This is used for continuous variable (Means)**

    <img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-16_at_12.25.59_PM.png" width="40%">

```sql
import statsmodels.stats.api as sms
effect_size = 0.1
alpha = 0.05 # significance level
power = 0.8
sample_size = sms.TTestIndPower().solve_power(effect_size = effect_size, power = power, alpha = alpha)
print('Required sample size ~ {0:.1f}'.format(sample_size) + ' per group')
#Output:
#Required sample size ~ 1570.7 per group
```

**Relationship between minimum sample size required vs minimum detectable difference**

- Find the balance or trade-off among enought power(not miss opportunity/fail to detect improvement)
- Minimum sample size (traffic volume) required.
- effective size (worth to do the test if the metric increase/decrease how much)- minimum detectable difference

    
    <img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/effect.png" width="40%">
    

    Sample Size vs Minimum Detectable Effective Size

```python
samplesize_list=[]
baseline=0.1
#set lift range: 5%~30% with 1% incrementality of baseline(0.1)
deltas=np.arange(0.005, 0.03, 0.001)
for delta in deltas:
  prob2=baseline_cvr+delta
  effect_size=sms.proportion_effectsize(baseline, prob2)
  sample_size=sms.NormalIndPower().solve_power(effect_size=effect_size, power=0.8, alpha=0.05, ratio=1)
  samplesize_list.append(sample_size)
plt.plot(deltas, samplesize_list)
plt.title('Minimum required sample size for minimum detectable delta/effective size')
plt.ylabel('Required Sample Size')
plt.xlabel('Minimum Detectable effective size')
plt.tight_layout()
plt.show()
```

### Collecting and Analyzing the data collected for the **control** and treatment group: **What to do once we have the data collected?**

- We want to ensure we run the test long enough to collect enough data as calculated in the previous step. Otherwise, it'll be hard to tell whether there was a statistically significant difference between the two variations
- Also, split of traffic not to be 50–50 and allocate more traffic to version A, in case you are concerned about losses due to version B. However, keep in mind that a very skewed split often leads to longer times before the A/B testing becomes *(statistically)* *significant*.

**Checks for invariant metrics:**

- Firstly we want to make sure all the invariant metrics does not show statistically  significant change between the two groups to ensure the study is not biased.
- This also ensures the data we have collected is right.

**Check for effect size on the Evaluation Metrics**

- The next step is looking at the changes between the control and experiment groups with regard to our evaluation metrics to make sure the difference is there, that it is statistically significant and most importantly practically significant (the difference is "big" enough to make the experimented change beneficial to the company.

### How do we know what test to conduct on each of these metrics we collected?

- We must  **plot the distributions of key variables**. In an RCT we have 2 or more groups **(e.g. Control and treatment)** to observe. Plotting density plots or boxplots for each key variable at the start of the analysis is helpful in determining the distribution of the data which is very **critical to the test we use becasue of the underlying assumptions.**
- We should also run a **Shapiro test** on our data to make sure it is normal *before* deciding what method to use (e.g. a T-test or a Wilcoxon test).
- Always report the p-value with **means and confidence intervals** for normal distributed data OR with **medians** and first and third quartiles for non-normal data.

The following sections on **"Data Distribution"** and "**Parametric and Non-Parametric Test"**  will help decide ****what static test to be used to measure each of the invariant metrics and evaluation metrics and draw conclusions about them at a chosen level of significance (alpha).

## **Data Distribution**

### Binomial Distribution **P(x success in N trials)**

**Only two possible outcomes**

**In order for a variable X to be a binomial random variable,**
• Each trial must be independent,
• Each trial can be called a “success” or “failure,”
• There are a fixed number of trials, and
• The probability of success on each trial is constant.

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-16_at_12.42.09_PM.png" width="40%">

**Probability Mass Function**

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-16_at_12.38.15_PM.png" width="40%">

**p= rate of success**

**x= number of success** 

**n=number of trials** 

**Code to create probablity Mass Function** 

```python
import matplotlib.pyplot as plt
from scipy.stats import binom
import numpy as np

# Determine the probability of having x number of click throughs
clicks = np.arange(20, 80)
num_a, num_b = 550, 450
click_a, click_b = 48, 56
rate_a, rate_b = click_a / num_a, click_b / num_b
prob_a = binom(num_a, rate_a).pmf(clicks)
prob_b = binom(num_b, rate_b).pmf(clicks)

# Make the bar plots.
plt.bar(clicks, prob_a, label="A", alpha=0.7)
plt.bar(clicks, prob_b, label="B", alpha=0.7)
plt.legend()
plt.xlabel("Num converted"); plt.ylabel("Probability");
plt.show()
```


<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/binomial.png" width="40%">


Probability Mass Function for Group A and Group B with given success rate

### **Normal Distribution**

**Application example: the amount of time a user spends on a website, can the ratings be used to recommend products to the user.** 

- The mean and median and mode  fall at the center of the ideal normal distribution
- aread under the curve is all the 1.0 and 100%
- normal distribution follows emperical rule 68-95-99.7
    - 2sigma covers 68% of the data
    - 4sigma covers 95% of the data
    - 6sigma covers 99.7% of the data
    - little tails 0.03 % of the data and equally divided on both sides.
- percentile 95th means 95% of the data lies below that curve
- Other forms of Gaussian is t-distribution/exponential distribution

**Z-score :** 

- how far is value from the mean in terms of standard deviations, in terms of percentile
- $z= (X-u)/sigma$ u is mean, X is data point, sigma is std deviation.
- for ex: if mean is 16 and the value we are looking for is 17.5, and std is 1, (17.5-16)/1 z-score is 1.5 then we look into the table it says this value z-table is 93.38% above all the values.
- If the value of z is 1.5 that means its 1.5 std above the mean and the area under the curve will always be the same.
- The z value and the z-table is used to identify the percentage of area under the curve, which is also the **p-value**
- z-score always give area **left to the curve**

**z-statistic**

- When **population standard deviation** **is known** we can use the z-score

     <img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/9EFFB6E2-024B-4DBB-B5DE-9065943FF75D.jpeg" width="40%">

- When **population standard deviation is unknown and we have sample greater  than 30 samples,** we use z-statistic for proportions only if we can assume the population is normally distributed, we use sample standard deviation in the formula, that can be proven by

    
     <img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/67898C9F-77A8-4CA7-B993-73D0E8B21F36.jpeg" width="40%">


       

- **z-test for comparing two proportions**

   
    <img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-18_at_10.37.00_AM.png" width="40%">

```sql
from statsmodels.stats.proportion import proportions_ztest
import pandas as pd
import numpy as np
X1, X2 = [486, 527]
n1, n2 = [5000, 5000]
conversions = np.array([X1, X2])
clicks = np.array([n1, n2])

zscore, pvalue = proportions_ztest(conversions, clicks, alternative = 'two-sided') #two-sided indicates two-tailed test
print('zscore = {:.4f}, pvalue = {:.4f}'.format(zscore, pvalue))
# [output]: zscore = -1.3589, pvalue = 0.1742
```

**Probability Density Function**


<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-16_at_9.57.05_PM.png" width="40%">

- It means how much probability is concentrated per unit length (d𝒙) near 𝒙, or how dense the probability is near 𝒙. x is any point on the x-axis and y is the ***P(x)***

The probability Density curve is not the same as the Probability of a function at (X=x). It is the integral of the probability density function.

**Probability Density Function with varying mean**

- What does this really mean?
    - It means if we have a  set of data say between `1000` linearly spaced points between `(-4 to 4)` and we plot a graph for `mean =1,2,3`  and keeping sigma constant at `1`.
    - The observation is the graph moves to the right which actually means the **density is moving towards the mean.**
    
    <img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-19_at_2.04.23_AM.png" width="40%">

**Probability Density Function with varying standard deviation**

- What does this really mean?
    - It means if we have a  set of data say between `1000` linearly spaced points between `(-10 to 10)` and we plot a graph for constant `mean =0` and keeping varying the sigma between `[2,4,6]`.
    - The observation is the graph gets broader which means the **probability** **density is spreading out.**
    
    <img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-19_at_1.58.49_AM.png" width="40%">

### Student’s T-distribution

- The T-distribution is used instead of the normal distribution when you have small samples (usually in practice less than 30).
- The larger the size of your sample, the more the t-distribution looks like the normal one. In fact, for sample sizes larger than 30 (e.g. more degrees of freedom), the distribution almost exactly follows the shape of the normal curve.

**Mean and Standard Deviation of t-distribution** 


<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-16_at_10.11.25_PM.png" width="40%">

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-16_at_10.10.57_PM.png" width="40%">

**t-distribution function**


<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-16_at_10.11.15_PM.png" width="40%">

**t-score**

- When **population standard deviation is unknown and we have a sample greater than 30 samples**, we use t-statistic, we use sample standard deviation in the formula.
- When **population standard deviation is unknown and we have sample size less  than 30 samples,** we use t-statistic only if we can assume the population is normally distributed,


 <img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-16_at_10.11.31_PM.png" width="40%">

### Chi-Square Distribution

- Chi-square distribution is a special case of gamma-distribution (just like T-distribution), and has only one parameter: degrees of freedom (ν), which is as simple as number of possible categories minus one.
- The distribution only has positive values, and it is right-screwed.
- Its shape varies depending on ν: from very asymmetric with low ν, to almost normally-shaped with very high ν(with ν approaches infinity, chi-square distribution becomes normal distribution)

**Mean and Standard Deviation of t-distribution**

- The mean of the distribution is equal to the number of degrees of freedom:
    
    <img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-16_at_11.30.37_PM.png" width="40%">

- The standard deviation is equal to the square root of two times the number of degrees of freedom:

    
     <img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-16_at_11.30.41_PM.png" width="40%">

**chi-square statistic**

- *Oi is the number of times i-category occurred in a sample*
- Ei is the assumption for the number of times i-category is should occur in a sample (expected frequency)

    
    <img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-16_at_11.33.19_PM.png" width="40%">


    Example is well explained here in this article: [https://towardsdatascience.com/the-ultimate-guide-to-a-b-testing-part-4-non-parametric-tests-4db7b4b6a974](https://towardsdatascience.com/the-ultimate-guide-to-a-b-testing-part-4-non-parametric-tests-4db7b4b6a974)

**The probability density function for chi2 is:**

k = degree of freedom

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-16_at_11.35.38_PM.png" width="40%">

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-20_at_5.33.06_PM.png" width="40%">

## **Parametric Test**

All statistical tests can be divided into two main groups: **parametric** and **non-parametric**. Both groups serve to solve more or less the same problems, but do it in a bit different situations: 

**Parametric tests** are used only when a normal (or close to normal) distribution is assumed. The most widely used tests are the **Z-test,** **t-test,** and **ANOVA.** Average session length is normally distributed, so we can use **parametric tests** to check the significance of the difference

### **Z test (Z statistic):**

- In a z-test, the sample is assumed to be **normally distributed**.
- A z-score is calculated with population parameters such as **“population mean” or  “population proportion” and “population standard deviation”** and is used to validate a hypothesis that the sample drawn belongs to the same population.

```python
def ztest_comparing_two_proportions(X1,X2,n1,n2):
    p1_hat = X1/n1
    p2_hat = X2/n2
    p_bar = (X1+X2)/(n1+n2)
    q_bar = (1-p_bar)
    z_diff= p1_hat-p2_hat
    z_sd = np.sqrt((1/n1+1/n2)*p_bar*q_bar)
    z_score = z_diff/z_stf
    p_value = norm().cdf(z_score)
    return z_diff, p_value,z_sd,z_score  

#X1 = count of success (1's or True or Win or Converted) in group A
#n1 = count of total samples in group A
#X2 = count of success (1's or True or Win or Converted) in group B
#n2 = count of total samples in group B
```

### **T-test (T statistic):**

- **A t-test is used when the population parameters (mean and standard deviation) are not known.**
- **T-test (T statistic):** A t-test is used to compare the mean of two given samples. Like a z-test, a t-test also assumes a normal distribution of the sample.

    There are three versions of t-test

    1. Independent samples t-test which compares mean for two groups
    2. Paired sample t-test which compares means from the same group at different times
    3. One sample t-test which tests the mean of a single group against a known mean.

```sql
from scipy.stats import ttest_ind
order_value_control_group = np.random.normal(0,1.11, 50)
order_value_experimental_group = np.random.normal(0,1.84, 32)
tscore, pval= **ttest_ind(order_value_control_group, order_value_experimental_group, equal_var=True)**
print(f"Zscore is {zscore:0.2f}, p-value is {prob:0.3f} (two tailed), {prob/2:0.3f} (one tailed)"
```

### **Welch test:**

- **Welch test:** IIf the data is normal, then we should always use the Welch test (and ignore the T-test) if we have  **unequal sample sizes and unequal variances**, which we will often come across in the real world than the t-test
- Welsch’s t-test is meant for continuous data
- If the data is discrete like  `0` and `1` options. A better option  is the Mann-Whitney U statistic.

```sql
from scipy.stats import ttest_ind
order_value_control_group = np.random.normal(0,1.11, 50)
order_value_experimental_group = np.random.normal(0,1.84, 32)
tscore, pval= **ttest_ind(order_value_control_group, order_value_experimental_group, equal_var=False)**
print(f"Zscore is {zscore:0.2f}, p-value is {prob:0.3f} (two tailed), {prob/2:0.3f} (one tailed)"
```

The following graphs shows how Welch test performs much better with **unequal sample sizes and variance**. the p-values in the left graph between 0-0.05, area of rejection is much higher where it should be only 5% based on alpha 0.05 hence it can lead to wrong results and inferences, where as welch test doesn't do that and much better suited for unuequal mean and variances. 

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/ttest.png" width="40%">

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/welch_test.png" width="40%">

- **ANOVA (F-statistic):** Similar to a T-test, ANOVA can tell you how significant the differences between groups are. While a t-test compares 2 groups, ANOVA test can do more than two groups.
    1. One-way ANOVA: It is used to compare the difference between the three or more samples/groups of a single independent variable.
    2. MANOVA: MANOVA allows us to test the effect of one or more independent variable on two or more dependent variables. In addition, MANOVA can also detect the difference in co-relation between dependent variables given the groups of independent variables.

## **Non-Parametric Test**

**Non-parametric tests** are used when continuous data is not normally distributed or when data is discrete. Some of the representatives are **chi-squared** and **Fisher’s exact tests, Mann–Whitney U-test. e.g.** Day-1 retention and conversion are binomial distributions (there are two outcomes for both cases: returned/churned, converted/didn’t convert), which means that we’ll need to use **non-parametric tests**

### **chi-squared test:**

- **chi-squared test:** This test works only for categorical data when you have two or more categories and want to check if there is a significant difference between them.
- A **chi-square goodness of fit test** determines if a sample data matches a population.
- A **chi-Square test for independence** compares two variables in a contingency table to see if they are related. In a more general sense, it tests to see whether distributions of categorical variables differ from each another:

### **Fisher’s exact test**

- **Fisher’s exact test:** Fisher’s exact test is a non-parametric  when the data set is small or categories are imbalanced instead of the chi-square test

    ```python
    from scipy.stats import fisher_exact
    oddsratio, pvalue = fisher_exact([[50, 2450], [42, 2458]])
    ```

Working is well explained here: [https://towardsdatascience.com/the-ultimate-guide-to-a-b-testing-part-4-non-parametric-tests-4db7b4b6a974](https://towardsdatascience.com/the-ultimate-guide-to-a-b-testing-part-4-non-parametric-tests-4db7b4b6a974)

Method I: use proportions test

```python
import statsmodels.stats.proportion as proportion
import numpy as np
converted = np.array([486, 527])
clicks = np.array([5000, 5000])
chisq, pvalue, table = proportion.proportions_chisquare(converted, clicks)
print('chisq =%.3f, pvalue = %.3f'%(chisq, pvalue))
print("Contingency Table:")
print(table)

```

Method II: Use contingency table — the traditional way

```python
import scipy.stats as stats
from scipy.stats import chi2
ob_table = np.array([[4514,  486], [4473,  527]])
result = stats.chi2_contingency(ob_table, correction = False)  # correction = False due to df=1
chisq, pvalue = result[:2]
print('chisq = {}, pvalue = {}'.format(chisq, pvalue))
```

### **Mann–Whitney U-test:**

- **Mann–Whitney U-test:**
- The Mann-Whitney U test is a nonparametric statistical significance test for determining whether two independent samples were drawn from a population with the same distribution.
- Mann-Whitney U test is commonly used to compare differences between two independent groups when the dependent variable is not normally distributed.
- Use only when the number of observation in each sample is `> 20` and you have `2` independent samples of ranks. Mann-Whitney U is significant if the u-obtained is LESS THAN or equal to the critical value of U.
- **The test is specifically for non-parametric distributions, which do not assume a specific distribution for a set of data. Because of this, the Mann-Whitney U Test can be applied to any distribution, whether it is Gaussian or not.**

```sql
import numpy as np
from scipy.stats import mannwhitneyu
sample1=[32, 34, 29, 39, 38, 37, 38, 36, 30, 26]
sample2=[40, 34, 30, 39, 38, 37, 38, 36, 50, 49]
stat, pvalue=mannwhitneyu(sample1, sample2)
print('statistics=%.3f, p=%.5f'%(stat,pvalue))
alpha=0.05
if pvalue> alpha:
  print('Two Groups are from the Same distribution(fail to reject H0) under alpha=0.05')
else:
  print('Two Groups are from Different distributions(reject H0) under alpha=0.05')
```

### **Wilcoxon Text**

- **Wilcoxon Text:** A **good test for non normal data** with very few assumptions.
- The Wilcoxon signed-rank test tests the null hypothesis that two related paired samples come from the same distribution. In particular, it tests whether the distribution of the differences x - y is symmetric about zero. It is a non-parametric version of the **paired T-test.**
- The main assumptions here are:
    - samples are randomly representative of population,
    - samples are independent of each other ,
    - values have an order (e.g. 3 is more than 1, but we can’t say that true is more than false).

```python
**#the differences in height between group A and group B  is given as follows:**
d = [6, 8, 14, 16, 23, 24, 28, 29, 41, -48, 49, 56, 60, -67, 75]
from scipy.stats import wilcoxon
w, p = wilcoxon(d)
w, p
Output: (24.0, 0.041259765625)
**we would reject the null hypothesis at a confidence level of 5%, concluding that there is a difference in height between the groups.**
```

## Drawing Conclusions

- The **minimum effect** we wanted to see is important in drawing conclusions about our test.
- Based on the significance level we picked (5%) we can define our **confidence interval**. When we picked the minimum effect, we're saying that we wanted to see at least that amount of difference between the results of the control and the experiment tests.
- For example, **the confidence interval for difference in distribution for the control and treatment group for the evaluation metric is between** `-0.39% and 0.08%`. Given the **minimum detectable level** defined in the practical significance level is `1%`, we could only **reject the null hypothesis if the confidence interval lower bound was above 1%.** Therefore we cannot reject the null hypothesis and conclude which landing page drives more conversions.
- If making a change is  much more expensive or requires more time and labour to incorporate into the current workflow, the recommendation will be  to **keep the original design**. Because even if it may be statistically significant but it will not be practically significant for the bsuiness.
- So we need to evaluate two things to recommend a change for a particular evaluatio metric
    - It should be **statistically significant**
    - It should be **practically significant** (i.e mimimum detectable difference should be  **above** lower bound of the confidence interval of the differenc between two distributions for control and treatment group)

**Code to calculate Confidence Interval for the difference between Two distributions and Margin of error** 

```sql
#Calculate pooled standard error and margin of error
se_pooled = math.sqrt(prob_pooled * (1 - prob_pooled) * (1 / total_users_control + 1 / total_users_treatment))
z_score = st.norm.ppf(1 - confidence_level / 2)
margin_of_error = se_pooled * z_score

#Calculate dhat, the estimated difference between probability of conversions in the experiment and control groups
d_hat = (conversions_treatment / total_users_treatment) - (conversions_control / total_users_control)

#Test if we can reject the null hypothesis
lower_bound = d_hat - margin_of_error
upper_bound = d_hat + margin_of_error

if practical_significance < lower_bound:
    print("Reject null hypothesis")
else: 
    print("Do not reject the null hypothesis")
    
print("The lower bound of the confidence interval is ", round(lower_bound * 100, 2), "%")
```

## Pros and Cons of AB Testing

**Pros A/B split testing:** A/B split tests have several advantages, including these:

- **Useful in low-data rate tests.** If your landing page has only a few conversions per day, you simply can’t use a more advanced tuning method.
- **Ease of implementation.** Many software packages support simple split tests. You even may be able to collect the data you need with your existing Web analytics tools.
- **Ease of test design.** Split tests don't have to be carefully designed or balanced. You simply decide how many versions you want to test and then split the available traffic evenly among them.
- **Ease of analysis.** Only very simple statistical tests are needed to determine the winners. All you have to do is compare the baseline version to each challenger to see if you’ve reached your desired statistical confidence level.
- **Flexibility in defining the variable values.** The ability to mix and match allows you to test a range of evolutionary and revolutionary alternatives in one test, without being constrained by the more granular definition of variables in a multivariate test.

**Cons A/B split testing:** A/B split tests also have their drawbacks, including these:

- **Limited number of recipes.** While you may want to test dozens of elements on your landing pages, because of the limited scope of split testing, you have to test your ideas one at a time.
- **Inefficient data collection.** Conducting multiple split tests back to back is the most wasteful kind of data collection. None of the information from a previous test can be reused to draw conclusions about the other variables you may want to test in the future.

## Future work

- Bayesian AB testing

## Resources

[https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/](https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/)

[https://towardsdatascience.com/ab-testing-in-real-life-9b490b3c50d1](https://towardsdatascience.com/ab-testing-in-real-life-9b490b3c50d1)

[https://towardsdatascience.com/introduction-to-statistics-e9d72d818745](https://towardsdatascience.com/introduction-to-statistics-e9d72d818745)

[https://cosmiccoding.com.au/tutorials/ab_tests](https://cosmiccoding.com.au/tutorials/ab_tests)

[https://towardsdatascience.com/the-ultimate-guide-to-a-b-testing-part-3-parametric-tests-2c629e8d98f8](https://towardsdatascience.com/the-ultimate-guide-to-a-b-testing-part-3-parametric-tests-2c629e8d98f8)

[https://towardsdatascience.com/power-analysis-made-easy-dfee1eb813a](https://towardsdatascience.com/power-analysis-made-easy-dfee1eb813a)

[https://towardsdatascience.com/statistical-tests-when-to-use-which-704557554740](https://towardsdatascience.com/statistical-tests-when-to-use-which-704557554740)

[https://towardsdatascience.com/the-art-of-a-b-testing-5a10c9bb70a4](https://towardsdatascience.com/the-art-of-a-b-testing-5a10c9bb70a4)

[https://towardsdatascience.com/determine-if-two-distributions-are-significantly-different-using-the-mann-whitney-u-test-1f79aa249ffb](https://towardsdatascience.com/determine-if-two-distributions-are-significantly-different-using-the-mann-whitney-u-test-1f79aa249ffb)

[https://towardsdatascience.com/python-code-from-hypothesis-test-to-online-experiments-with-buiness-cases-e0597c6d1ec](https://towardsdatascience.com/python-code-from-hypothesis-test-to-online-experiments-with-buiness-cases-e0597c6d1ec)

[https://medium.com/@robbiegeoghegan/implementing-a-b-tests-in-python-514e9eb5b3a1](https://medium.com/@robbiegeoghegan/implementing-a-b-tests-in-python-514e9eb5b3a1)

[https://www.evanmiller.org/ab-testing/chi-squared.html](https://www.evanmiller.org/ab-testing/chi-squared.html)

[https://github.com/baumanab/udacity_ABTesting#summary](https://github.com/baumanab/udacity_ABTesting#summary)

[https://medium.com/@moggirain/a-complete-guide-about-a-b-testing-a1830410a0db](https://medium.com/@moggirain/a-complete-guide-about-a-b-testing-a1830410a0db)

[https://towardsdatascience.com/introduction-to-statistics-e9d72d818745](https://towardsdatascience.com/introduction-to-statistics-e9d72d818745)
