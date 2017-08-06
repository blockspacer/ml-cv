library(fpp)

# ARIMA models

# "ARIMA models provide another approach to time series 
# forecasting. Exponential smoothing and ARIMA models are 
# the two most widely-used approaches to time series 
# forecasting, and provide complementary approaches 
# to the problem. "

# exp use thrend
# arima - autocorrelation in data

# "Stationarity and differencing"

plot(diff(log(a10), 12))


# Unit root test
adf.test(diff(log(a10), 12), alternative = 'stationary')


# Autoregression model
# use own out? kind of IIR

# Moving av model (!! not smoothing)
# MA(q)

# non seasonal arima

fit <- auto.arima(usconsumption[,1], seasonal = FALSE)
plot(forecast(fit, h=10), include = 80)
par(mfrow=c(1, 2))
Acf(usconsumption[,1])
Pacf(usconsumption[,1])

# after we estimated p, d and q
# need phi and theta - MLE

# ARIMA much harder to estimate then regression


