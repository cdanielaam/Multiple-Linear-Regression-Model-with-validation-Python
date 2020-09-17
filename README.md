# Multiple-Linear-Regression-Model-with-validation-Python
Welcome to my Python project!

This is my first Python project and it is a predictive model for spirometry reference equations in Portuguese children with model validation. Two chorts were used, one cohort 
(derivation cohort) was used to create the predicive model, and the model was validated in a second cohort of childre (validation cohort). Unfortnatly I cannot share 
database with you, but feel free to use the code and apply it to your own analysis. You can also reach me if you detect any error in my code, if you have any sugestion, 
if you have doubts or if you believe I can help you in someway.

The steps and procedures used in code are documented and described bellow:

#1: Descriptive statistics for derivation cohort:
      Mean (SD) for age, height, weight and BMI, as well as spirometry parameters;
      Two-sample T test comparrisson between boys and girls;
  
#2: Descriptive statistics for validation cohort:
      Mean (SD) for age, height, weight and BMI, as well as spirometry parameters;
      Two-sample T test comparrisson between boys and girls;
    
#3: Comparisson between derivation and validation cohort (Two sample T-Test);

#4: Levene's Test of variance comparing antropometric and lung function parameters in both cohorts;

#5: Pearson's correlation coeficient between antropometric parameters and lung function parameters in the derivation cohort with scatterplots:
    1) Including all participants;
    2) Including only male participants;
    3) Including only female participants;
    
#6: Multiple Linear Regression models built and tested using antropometric parameters with model summary, coefficient intervals for all variables and tested 
for residuals (tested for both samples, derivation and validadtion). The model that incldes all variables was chosen. Model was chosen even if variables were 
not significant to increase model accuracy;

#7: Model was tested using Bland-ALtaman graph and statistics, and compared to GLI2012 predictive equations using mean differences;

#8: Correlations coefficients and scaterplots between ARIA predicted equations and measured values, and GLI2012 predictive equations and measured values.
