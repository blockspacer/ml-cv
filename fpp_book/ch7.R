library(fpp)

# exp. smooth.
# SES
# no thrend or seasonal patterns. It's important

oildata <- window(oil, start=1996, end=2007)
plot(oildata)

# selesction
fit1 <- ses(oildata, alpha=0.2, initial="simple", h=3)
fit2 <- ses(oildata, alpha=0.6, initial="simple", h=3)
fit3 <- ses(oildata, h=3)
plot(fit1, plot.conf=FALSE, ylab="Oil (millions of tonnes)",
     xlab="Year", main="", fcol="white", type="o")
lines(fitted(fit1), col="blue", type="o")
lines(fitted(fit2), col="red", type="o")
lines(fitted(fit3), col="green", type="o")
lines(fit1$mean, col="blue", type="o")
lines(fit2$mean, col="red", type="o")
lines(fit3$mean, col="green", type="o")
legend("topleft",lty=1, col=c(1,"blue","red","green"), 
       c("data", expression(alpha == 0.2), expression(alpha == 0.6),
         expression(alpha == 0.89)),pch=1)

# Thought: shift important if estimate thrend?

# Holt's linear thrend method
air <- window(ausair, start=1990, end = 2004)
fit1 <- holt(air, alpha = 0.8, beta = 0.2, initial = 'simple',
             h=5)
fit2 <- holt(air, alpha = 0.8, beta = 0.2, initial = 'simple',
             exponential = TRUE, h=5)

fit1$model$states
fitted(fit1)  # fixme: what is mean
fit1$mean

# fit3
# about "simple" was warning
fit3 <- holt(air, alpha = 0.8, beta = 0.2, initial = 'optimal',
             damped = TRUE, h=5)
plot(fit2, type = "o", fcol = "white")#, plot.conf = FALSE)
lines(fitted(fit1), col="blue")
lines(fitted(fit2), col="red")
lines(fitted(fit3), col="green")

lines(fit1$mean, col='blue', type='o')
lines(fit2$mean, col='red', type='o')
lines(fit3$mean, col='green', type='o')

# Damped trend methods

# 7.7 Innovations state space model for exp. smooth.


