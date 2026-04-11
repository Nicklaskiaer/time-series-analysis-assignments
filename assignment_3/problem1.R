# Set seed for reproducibility
set.seed(123)

# Number of observations
n <- 200

# Number of realizations
n_sim <- 5

phi1 <- -0.6
phi2 <- 0.5


# Simulate 5 realizations of the AR(2) process
# Model in assignment: X_t + phi1*X_{t-1} + phi2*X_{t-2} = e_t
# In arima.sim(), AR signs are flipped, so we use ar = c(-phi1, -phi2)
sim_data <- replicate(
  n_sim,
  arima.sim(n = n, model = list(ar = c(-phi1, -phi2)))
)

png("assignment_3/timeseries.png", width = 1200, height = 600)

# Plot all 5 realizations in one figure
matplot(
  sim_data,
  type = "l",        # line plot
  lty = 1,           # solid lines
  xlab = "Time",
  ylab = expression(X[t]),
  main = "5 realizations of the AR(2) process"
)

# Add legend
legend(
  "topright",
  legend = paste("Realization", 1:5),
  lty = 1,
  col = 1:5,
  bty = "n"
)

dev.off()


##############################################################################################################
# problem 1.2 

# Number of lags
lag_max <- 30

# --- Theoretical ACF ---
# Compute theoretical ACF using ARMAacf
acf_theoretical <- ARMAacf(ar = c(-phi1, -phi2), lag.max = lag_max)

# --- Empirical ACF ---
# Compute ACF for each of the 5 simulations
acf_empirical <- apply(sim_data, 2, function(x) {
  acf(x, plot = FALSE, lag.max = lag_max)$acf
})

# --- Plot ---
# Plot first empirical ACF
plot(
  0:lag_max,
  acf_empirical[,1],
  type = "l",
  ylim = range(acf_empirical, acf_theoretical),
  xlab = "Lag",
  ylab = "ACF",
  main = "Empirical vs Theoretical ACF"
)

# Add remaining empirical ACFs
for(i in 2:ncol(acf_empirical)){
  lines(0:lag_max, acf_empirical[,i], col = i)
}

# Add theoretical ACF (thick black line)
lines(0:lag_max, acf_theoretical, lwd = 3, col = "black")

# Legend
legend(
  "topright",
  legend = c("Empirical (5 sims)", "Theoretical"),
  col = c("gray", "black"),
  lwd = c(1,3),
  bty = "n"
)



##############################################################################################################

# problem 1.3
set.seed(123)

# Number of observations
n <- 200

# Number of realizations
n_sim <- 5

phi1 <- -0.6
phi2 <- -0.3

sim_data <- replicate(
  n_sim,
  arima.sim(n = n, model = list(ar = c(-phi1, -phi2)))
)

# Number of lags
lag_max <- 30



# --- Theoretical ACF ---
# Compute theoretical ACF using ARMAacf
acf_theoretical <- ARMAacf(ar = c(-phi1, -phi2), lag.max = lag_max)

# --- Empirical ACF ---
# Compute ACF for each of the 5 simulations
acf_empirical <- apply(sim_data, 2, function(x) {
  acf(x, plot = FALSE, lag.max = lag_max)$acf
})

# --- Plot ---
# Plot first empirical ACF
plot(
  0:lag_max,
  acf_empirical[,1],
  type = "l",
  ylim = range(acf_empirical, acf_theoretical),
  xlab = "Lag",
  ylab = "ACF",
  main = "Empirical vs Theoretical ACF"
)

# Add remaining empirical ACFs
for(i in 2:ncol(acf_empirical)){
  lines(0:lag_max, acf_empirical[,i], col = i)
}

# Add theoretical ACF (thick black line)
lines(0:lag_max, acf_theoretical, lwd = 3, col = "black")

# Legend
legend(
  "topright",
  legend = c("Empirical (5 sims)", "Theoretical"),
  col = c("gray", "black"),
  lwd = c(1,3),
  bty = "n"
)


##############################################################################################################

# problem 1.4
set.seed(123)

# Number of observations
n <- 200

# Number of realizations
n_sim <- 5

phi1 <- 0.6
phi2 <- -0.3

sim_data <- replicate(
  n_sim,
  arima.sim(n = n, model = list(ar = c(-phi1, -phi2)))
)

# Number of lags
lag_max <- 30



# --- Theoretical ACF ---
# Compute theoretical ACF using ARMAacf
acf_theoretical <- ARMAacf(ar = c(-phi1, -phi2), lag.max = lag_max)

# --- Empirical ACF ---
# Compute ACF for each of the 5 simulations
acf_empirical <- apply(sim_data, 2, function(x) {
  acf(x, plot = FALSE, lag.max = lag_max)$acf
})

# --- Plot ---
# Plot first empirical ACF
plot(
  0:lag_max,
  acf_empirical[,1],
  type = "l",
  ylim = range(acf_empirical, acf_theoretical),
  xlab = "Lag",
  ylab = "ACF",
  main = "Empirical vs Theoretical ACF"
)

# Add remaining empirical ACFs
for(i in 2:ncol(acf_empirical)){
  lines(0:lag_max, acf_empirical[,i], col = i)
}

# Add theoretical ACF (thick black line)
lines(0:lag_max, acf_theoretical, lwd = 3, col = "black")

# Legend
legend(
  "topright",
  legend = c("Empirical (5 sims)", "Theoretical"),
  col = c("gray", "black"),
  lwd = c(1,3),
  bty = "n"
)


##############################################################################################################

# problem 1.5 den ene rod kommer til ligge på enhedscirklen, så vi har en ikke-stationær proces. Vi kan stadig simul

##############################################################################################################
# problem 1.6 (theoretical only)

set.seed(123)

phi1 <- -0.75
phi2 <- -0.3

lag_max <- 30

# --- Theoretical ACF ---
acf_theoretical <- ARMAacf(ar = c(-phi1, -phi2), lag.max = lag_max)

# --- Plot ---
plot(
  0:lag_max,
  acf_theoretical,
  type = "l",
  lwd = 3,
  col = "black",
  xlab = "Lag",
  ylab = "ACF",
  main = "Theoretical ACF"
)


################
# Set seed for reproducibility
set.seed(123)

# Number of observations
n <- 200

# Number of realizations
n_sim <- 5

phi1 <- -0.75
phi2 <- -0.3

# Storage matrix
sim_data <- matrix(0, nrow = n, ncol = n_sim)

# Simulate 5 realizations recursively
for (j in 1:n_sim) {
  
  # White noise
  eps <- rnorm(n, mean = 0, sd = 1)
  
  # Initial values
  x <- numeric(n)
  x[1] <- eps[1]
  x[2] <- 0.75 * x[1] + eps[2]
  
  # Recursive simulation
  for (t in 3:n) {
    x[t] <- 0.75 * x[t-1] + 0.3 * x[t-2] + eps[t]
  }
  
  sim_data[, j] <- x
}

# Plot all 5 realizations in one figure
matplot(
  sim_data,
  type = "l",
  lty = 1,
  xlab = "Time",
  ylab = expression(X[t]),
  main = "5 realizations of the AR(2) process"
)

# Add legend
legend(
  "topleft",
  legend = paste("Realization", 1:5),
  lty = 1,
  col = 1:5,
  bty = "n"
)