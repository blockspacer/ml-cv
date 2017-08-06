library("fpp")

# https://www.r-bloggers.com/seasonal-trend-decomposition-in-r/
# 2 - trend, 3 - remainder
#str(elecequip)
#str(fit)
fit <- stl(elecequip, s.window = 5)

# что-то оооочень большое и похоже хочет какой-то автоизации
#install.packages("plotly")
# Plotting
par(mfrow=c(2,1))
plot(elecequip, col="gray", main="eem", ylab="noi", xlab="")
# https://stackoverflow.com/questions/12389318/dollar-sign-before-a-variable
#The $ allows you extract elements by name from a named list. For example
#x <- list(a=1, b=2, c=3)
#x$b
# [1] 2
lines(fit$time.series[,2], col='red', ylab='trend')

# Next
plot(fit$time.series[,3], col='green', ylab='trend')
#lines(fit$time.series[,1], col='green', ylab='trend')

# All together
#plot(fit)
monthplot(fit$time.series[,'seasonal'], main="", ylab="")

# fixme: combine plots
# http://www.statmethods.net/advgraphs/layout.html

# Seasonally adj. data
# yt - St
plot(elecequip, col="gray", main="eem", ylab="noi", xlab="")
lines(seasadj(fit))


# New
#install.packages("dplyr")

############################################################
# 6.2 Moving avg.
# series decomposition
ma(elecsales, order = 5)

par(mfrow=c(1,1))
plot(elecsales, main="", ylab='GWh', xlab="Year")
lines(ma(elecsales, 5), col='red')

# "Moving averages of moving averages"
# Weighted moving avg
# Attengion!!! Phase shift
# http://www.abs.gov.au/websitedbs/d3310114.nsf/51c9a3d36edfd0dfca256acb00118404/5fc845406def2c3dca256ce100188f8e!OpenDocument

# classical
# "While classical decomposition is still widely used, it is not recommended."
#fit <- decompose(x, type='multiplicative')
# 1. dead zone
# 

# X-12-ARIMA
# Q: x12 - типа за год, по месяцам?
# "One of the most popular methods for decomposing quarterly 
# and monthly data is X-12-ARIMA,"
# "The “ARIMA” part of X-12-ARIMA refers to 
# the use of an ARIMA model (see Chapter 7) that provides 

# forecasts of the series forward in time, 
# (!!!!)as well as backwards in time."

# STL

# Summary
# decomp - for study, forecusting...

fit <- stl(elecequip, t.window = 15, 
           s.window = "periodic", robust = TRUE)

eeadj <-seasadj(fit)
plot(naive(eeadj))  # looks bad for me

fcast <- forecast(fit, method = 'naive')
plot(fcast)

#####################################################
# Attention!!! big another example
# http://ucanalytics.com/blogs/forecasting-time-series-analysis-manufacturing-case-study-example-part-1/
# Arima example
# http://ucanalytics.com/blogs/step-by-step-graphic-guide-to-forecasting-through-arima-modeling-in-r-manufacturing-case-study-example/

# Attention!!!
# Online cource "Applied Time Series Analysis"
# https://onlinecourses.science.psu.edu/stat510/node/41