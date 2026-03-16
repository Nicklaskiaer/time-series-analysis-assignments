rho <- ARMAacf(ar = c(-0.7, -0.2), lag.max = 30)

plot(0:30, rho,
     type = "h", lwd = 2,
     xlab = "Lag k", ylab = expression(rho(k)),
     main = "Autocorrelation for AR(2)")
abline(h = 0)
points(0:30, rho, pch = 19)