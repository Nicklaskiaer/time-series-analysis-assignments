# Problem 4: WLS - Local Linear Trend Model

# Read data from read_data.R
source("read_data.R")

# ------------------------------------------------------------------------------------
# 4.1 Describe the variance-covariance matrix (the N × N matrix Σ 
# (i.e. 72 × 72 matrix, so present only relevant parts of it)) for
# the local model and compare it to the variance-covariance matrix
# of the corresponding global model.

# We set up the variance-covariance matrix Sigma in the form given on slide 14 / 36 of 
# lecture 04, and this Sigma = Diag(1/lambda^(N-1),...,1/lambda,1). We also use the 
# provided value for lambda = 0.9

lambda <- 0.9
N <- nrow(Dtrain)
# chronological order
Dtrain <- Dtrain[order(Dtrain$time), ]
# set up weights from oldest to newest
w <- lambda^((N-1):0)

# Contruct Sigma matrix: 
Sigma <- diag(1/w)

# ------------------------------------------------------------------------------------
# 4.2 . Plot the ”λ-weights” vs. time in order to visualise how the training data 
# is weighted. Which time-point has the highest weight?
plot(Dtrain$time, w, type="b", xlab="Time", ylab="Lambda-weights", main="Lambda-weights vs Time")

# print last time point and its weight
last_time_point <- Dtrain$time[N]
last_weight <- w[N]
print(last_time_point)
print(last_weight)
# > print(last_time_point)
# [1] "2023-12-01 UTC"
# > print(last_weight)
# [1] 1

# ------------------------------------------------------------------------------------
# 4.3  Also calculate the sum of all the λ-weights. What would be the 
# corresponding sum of weights in an OLS model?
sum_weights <- sum(w)
print(sum_weights)
# > print(sum_weights)
# [1] 9.994925

# ------------------------------------------------------------------------------------
# 4.4 Estimate and present θ_1 and θ_2 corresponding to the WLS model with λ = 0.9.
# Response variable: Dtrain$total
y <- Dtrain$total
# Design matrix: intercept and year 
X <- cbind(1, Dtrain$year)

# Estimate theta using the WLS formula: theta = (X^T Sigma^{-1} X)^(-1) X^T Sigma^{-1} y
theta <- solve(t(X) %*% solve(Sigma) %*% X) %*% t(X) %*% solve(Sigma) %*% y
print(theta)
#             [,1]
# [1,] -52.4828617
# [2,]   0.0275299
# > 



# ------------------------------------------------------------------------------------
# 4.5 Make a forecast for the next 12 months - i.e., compute predicted values corresponding to the
# WLS model with λ = 0.9.

# Chronological order for test set
Dtest <- Dtest[order(Dtest$time), ]

# OLS fit (global model)
fit_ols <- lm(total ~ year, data = Dtrain)

# WLS fit (local model)
fit_wls <- lm(total ~ year, data = Dtrain, weights=w)

# Forecast for testperiod
pred_ols <- predict(fit_ols, newdata = Dtest, interval = "prediction", level = 0.95)
pred_wls <- predict(fit_wls, newdata = Dtest, interval = "prediction", level = 0.95)

# Forecast table
forecast_table <- data.frame(
  time = Dtest$time,
  year = Dtest$year,
  y_obs = Dtest$total,
  ols_fit = pred_ols[, "fit"],
  ols_lwr = pred_ols[, "lwr"],
  ols_upr = pred_ols[, "upr"],
  wls_fit = pred_wls[, "fit"],
  wls_lwr = pred_wls[, "lwr"],
  wls_upr = pred_wls[, "upr"]
)

########################

# 1) Vælg x-interval: sidste del af train + hele test + lidt "luft" frem
x_left  <- as.POSIXct("2023-01-01", tz="UTC")  # justér hvis du vil se mere/mindre
x_right <- max(Dtest$time) + 35*24*3600        # ~35 dage ekstra til højre

train_win <- Dtrain[Dtrain$time >= x_left & Dtrain$time <= x_right, ]

# 2) Vælg y-interval ud fra det der faktisk er i vinduet (inkl. PI)
y_all <- c(
  train_win$total,
  Dtest$total,
  pred_ols[, c("fit","lwr","upr")],
  pred_wls[, c("fit","lwr","upr")]
)
ylim <- range(y_all, na.rm = TRUE)

# (valgfrit) lidt luft op/ned
pad <- 0.02 * diff(ylim)
ylim <- c(ylim[1] - pad, ylim[2] + pad)

# 3) Plot: først træning
plot(train_win$time, train_win$total, pch = 16, col = "black",
     xlab = "Time", ylab = "Total (millions)",
     main = "OLS vs WLS: 12-month forecast with prediction intervals",
     xlim = c(x_left, x_right), ylim = ylim)

# 4) Plot test-observationer
points(Dtest$time, Dtest$total, pch = 16, col ="red3")

# 5) OLS: fit + prediction interval
lines(Dtest$time, pred_ols[, "fit"], col = "dodgerblue3", lwd = 2)
lines(Dtest$time, pred_ols[, "lwr"], col = "dodgerblue3", lwd = 1, lty = 2)
lines(Dtest$time, pred_ols[, "upr"], col = "dodgerblue3", lwd = 1, lty = 2)

# 6) WLS: fit + prediction interval
lines(Dtest$time, pred_wls[, "fit"], col = "green", lwd = 2)
lines(Dtest$time, pred_wls[, "lwr"], col = "green", lwd = 1, lty = 3)
lines(Dtest$time, pred_wls[, "upr"], col = "green", lwd = 1, lty = 3)

# 7) Legend
legend("bottomright",
       legend = c("Train obs", "Test obs",
                  "OLS forecast", "OLS prediction interval",
                  "WLS forecast", "WLS prediction interval"),
       col    = c("black", "red3",
                  "dodgerblue3", "dodgerblue3",
                  "green", "green"),
       pch    = c(16, 16, NA, NA, NA, NA),
       lty    = c(NA, NA, 1, 2, 1, 3),
       lwd    = c(NA, NA, 2, 1, 2, 1),
       bty    = "n")




# === Second plot: full history (train + test) with OLS/WLS forecasts and PIs ===

# 1) Choose x-range: full training period + full test period + some space to the right
x_left_full  <- min(Dtrain$time)
x_right_full <- max(Dtest$time) + 35*24*3600  # ~35 days extra space on the right

# 2) Choose y-range based on all observations and prediction intervals
y_all_full <- c(
  Dtrain$total,
  Dtest$total,
  pred_ols[, c("fit","lwr","upr")],
  pred_wls[, c("fit","lwr","upr")]
)
ylim_full <- range(y_all_full, na.rm = TRUE)

# (optional) add a small padding
pad_full <- 0.02 * diff(ylim_full)
ylim_full <- c(ylim_full[1] - pad_full, ylim_full[2] + pad_full)

# 3) Plot full training data
plot(Dtrain$time, Dtrain$total, pch = 16, col = "black",
     xlab = "Time", ylab = "Total (millions)",
     main = "OLS vs WLS: Full history with 12-month forecasts",
     xlim = c(x_left_full, x_right_full), ylim = ylim_full)

# 4) Add test observations
points(Dtest$time, Dtest$total, pch = 16, col = "red3")

# 5) Add OLS forecast + prediction interval (test set)
lines(Dtest$time, pred_ols[, "fit"], col = "dodgerblue3", lwd = 2)
lines(Dtest$time, pred_ols[, "lwr"], col = "dodgerblue3", lwd = 1, lty = 2)
lines(Dtest$time, pred_ols[, "upr"], col = "dodgerblue3", lwd = 1, lty = 2)

# 6) Add WLS forecast + prediction interval (test set)
lines(Dtest$time, pred_wls[, "fit"], col = "green", lwd = 2)
lines(Dtest$time, pred_wls[, "lwr"], col = "green", lwd = 1, lty = 3)
lines(Dtest$time, pred_wls[, "upr"], col = "green", lwd = 1, lty = 3)

# 7) Legend
legend("topleft",
       legend = c("Train obs", "Test obs",
                  "OLS forecast", "OLS prediction interval",
                  "WLS forecast", "WLS prediction interval"),
       col    = c("black", "red3",
                  "dodgerblue3", "dodgerblue3",
                  "green", "green"),
       pch    = c(16, 16, NA, NA, NA, NA),
       lty    = c(NA, NA, 1, 2, 1, 3),
       lwd    = c(NA, NA, 2, 1, 2, 1),
       bty    = "n")


# === Second plot: full history (train + test) with OLS/WLS forecasts and PIs ===

# 1) Choose x-range: full training period + full test period + some space to the right
x_left_full  <- min(Dtrain$time)
x_right_full <- max(Dtest$time) + 35*24*3600  # ~35 days extra space on the right

# 2) Choose y-range based on all observations and prediction intervals
y_all_full <- c(
  Dtrain$total,
  Dtest$total,
  pred_ols[, c("fit","lwr","upr")],
  pred_wls[, c("fit","lwr","upr")]
)
ylim_full <- range(y_all_full, na.rm = TRUE)

# (optional) add a small padding
pad_full <- 0.02 * diff(ylim_full)
ylim_full <- c(ylim_full[1] - pad_full, ylim_full[2] + pad_full)

# 3) Plot full training data
plot(Dtrain$time, Dtrain$total, pch = 16, col = "black",
     xlab = "Time", ylab = "Total (millions)",
     main = "OLS vs WLS: Full history with 12-month forecasts",
     xlim = c(x_left_full, x_right_full), ylim = ylim_full)

# 4) Add test observations
points(Dtest$time, Dtest$total, pch = 16, col = "red3")

# 5) Add OLS forecast + prediction interval (test set)
lines(Dtest$time, pred_ols[, "fit"], col = "dodgerblue3", lwd = 2)
lines(Dtest$time, pred_ols[, "lwr"], col = "dodgerblue3", lwd = 1, lty = 2)
lines(Dtest$time, pred_ols[, "upr"], col = "dodgerblue3", lwd = 1, lty = 2)

# 6) Add WLS forecast + prediction interval (test set)
lines(Dtest$time, pred_wls[, "fit"], col = "green", lwd = 2)
lines(Dtest$time, pred_wls[, "lwr"], col = "green", lwd = 1, lty = 3)
lines(Dtest$time, pred_wls[, "upr"], col = "green", lwd = 1, lty = 3)

# 7) Legend
legend("topleft",
       legend = c("Train obs", "Test obs",
                  "OLS forecast", "OLS prediction interval",
                  "WLS forecast", "WLS prediction interval"),
       col    = c("black", "red3",
                  "dodgerblue3", "dodgerblue3",
                  "green", "green"),
       pch    = c(16, 16, NA, NA, NA, NA),
       lty    = c(NA, NA, 1, 2, 1, 3),
       lwd    = c(NA, NA, 2, 1, 2, 1),
       bty    = "n")
