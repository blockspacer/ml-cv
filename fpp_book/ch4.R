library(fpp)

# least square estimator

# "So correlation and regression are strongly linked. 
# The advantage of a regression model over correlation 
# is that it asserts a predictive relationship between 
# the two variables (xx predicts yy) and quantifies this 
# in a way that is useful for forecasting."

ls("package:fpp")
#search()

# jitter add some jitter
# https://stat.ethz.ch/R-manual/R-devel/library/base/html/tilde.html
# Hmm...
# carbon = f(city) ?
# https://cran.r-project.org/doc/contrib/Lemon-kickstart/kr_form.html
plot(jitter(Carbon) ~ jitter(City), data=fuel)
fit <- lm(Carbon ~ City, data = fuel)

# can add any line
# http://www.sthda.com/english/wiki/abline-r-function-an-easy-way-to-add-straight-lines-to-a-plot-using-r-software
abline(fit)

summary(fit)


# evaluate
res <- residuals(fit)

# have non random pattern
# "A non-random pattern may indicate that a non-linear 
#relationship may be required, or some heteroscedasticity 
#is present (i.e., the residuals show non-constant variance), 
#or there is some left over serial correlation (only when 
#the data are time series)."

plot(jitter(res) ~ jitter(City), data=fuel)
abline(0,0)

# outliers
# "One source for an outlier occurring is an 
# incorrect data entry. Simple descriptive statistics 
# of your data can identify minima and maxima that are 
# not sensible. If such an observation is identified, 
# and it has been incorrectly recorded, it should be 
# immediately removed from the sample."

# "It is wise to report results both with and 
# without the removal of such observations."


# Goodness-of-fit
# 1. R**2 не всегдя отражает 
# 2. standart error of regression - but have troubles

# forecasting
fitted(fit)[1]
fcast <- forecast(fit, newdata = data.frame(City=30))
plot(fcast)





