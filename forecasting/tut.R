
# http://mazamascience.com/WorkingWithData/?p=728
# install.packages(c("fpp"))

#install.packages(c("forecast"), c("fma"), c("expsmooth"), c("lmtest"), c("tseries"))

# https://stackoverflow.com/questions/29094204/why-cant-i-install-forecast-package-in-rstudio
# сингапур
#install.packages(c("forecast"))
#install.packages(c("ggplot2"))
# R 3.1....
# https://askubuntu.com/questions/614530/how-to-install-latest-version-of-r-on-ubuntu-12-04-lts

# remove old!!!

library('tools')
#installFoundDepends(pkgDepends('forecast', local = FALSE)$Found)
#intall('forecast', dependencies = TRUE)
#"fma", c("expsmooth"), c("lmtest"), c("tseries"), 'fpp'
# install.packages("fma", dependencies=TRUE, repos='http://cran.us.r-project.org')
#install.packages("fpp", dependencies=TRUE, repos='http://cran.us.r-project.org')
#install.packages("tseries", dependencies=TRUE, repos='http://cran.us.r-project.org')

#library(help="tseries")

library(fpp)

#X11()
plot(a10)

# fixme: лучше в rstudio
#message("Press Return To Continue")
#invisible(readLines("stdin", n=1))


# rstudio - не работает связь между рабочей директорией и навигацией
#   по дереву каталогов
tute1 = read.csv("tute1.csv")
head(tute1)
tail(tute1)

# indexing with 1
tute1[, 1]
dim(tute1)  # rows, cols

# data to time series
tute_ts <- ts(tute1[, -1], start=1981, frequency = 4)

plot(tute_ts)

# TODO: interpolation
xs <- c(-1., 0., 1)
ys <- c(1.83, 0.2, 2)

x1 = xs[1]
x2 = xs[2]
x3 = xs[3]

y1 = ys[1]
y2 = ys[2]
y3 = ys[3]

# порядок образование матницы по столбцам, а не строкам
A = matrix(
  c(x1, x1**2, 1, x2, x2**2, 1, x3, x3**2, 1),
  nrow = 3,
  ncol = 3
)

A <- t(A)
A
Y = matrix(c(y1, y2, y3), nrow = 3, ncol=1) 

#B = matrix( 
#     c(2, 4, 3, 1, 5, 7), 
#     nrow=3, 
#     ncol=2)

#B <- t(A)
B <- solve(A)

C <- as.matrix(B) %*% as.matrix(Y)
xs_new <- seq.int(x1, x3, 0.001)
C


# http://www.johnmyleswhite.com/notebook/2010/09/23/higher-order-functions-in-r/
valf <- function(x)
{
  A
  A %*% C
  return (C[1,1] * x + x**2 * C[2, 1] + C[3, 1])
}
valf(-1)

G <- Map(valf, xs_new)

# plot
plot(xs, ys, type='l')
lines(xs_new, G, type='l', col='green')


