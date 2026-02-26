# ----------------------- 
# Problem 5.1: First two RLS iterations (compute R1, R2 and optionally theta1, theta2)

# Make sure training data is in chronological order
Dtrain <- Dtrain[order(Dtrain$time), ]

# Extract first two observations
y1 <- Dtrain$total[1]
y2 <- Dtrain$total[2]

x1 <- matrix(c(1, Dtrain$year[1]), ncol = 1)  # [1; year_1]
x2 <- matrix(c(1, Dtrain$year[2]), ncol = 1)  # [1; year_2]

# Initial values given in the assignment
R0 <- diag(c(0.1, 0.1))
theta0 <- matrix(c(0, 0), ncol = 1)

# ---- Iteration t = 1 ----
R1 <- R0 + x1 %*% t(x1)

# parameter update
theta1 <- theta0 + solve(R1) %*% x1 %*% (y1 - t(x1) %*% theta0)

# ---- Iteration t = 2 ----
R2 <- R1 + x2 %*% t(x2)

# Optional: parameter update
theta2 <- theta1 + solve(R2) %*% x2 %*% (y2 - t(x2) %*% theta1)

# Print results 
cat("R1 =\n"); print(R1)
cat("\nR2 =\n"); print(R2)

cat("\n(Optional) theta1 =\n"); print(theta1)
cat("\n(Optional) theta2 =\n"); print(theta2)

# -------------------------------------------------------------------------------------
# 5.2 for loop implementation of RLS

# Ensure training data is ordered
Dtrain <- Dtrain[order(Dtrain$time), ]

# Initial values given in the assignment
R <- diag(c(0.1, 0.1))
theta_hat <- matrix(c(0, 0), ncol = 1)

# Store results for presentation
res <- data.frame(
  t = integer(),
  time = as.POSIXct(character()),
  theta1 = numeric(),
  theta2 = numeric()
)

# Run RLS updates for t = 1..3
for (t in 1:3) {
  # Current regressor vector x_t = [1; year_t]
  x_t <- matrix(c(1, Dtrain$year[t]), ncol = 1)
  
  # Current observation y_t
  y_t <- Dtrain$total[t]
  
  # Update information matrix R_t
  R <- R + x_t %*% t(x_t)
  
  # Prediction error (innovation): y_t - x_t^T theta_{t-1}
  err <- as.numeric(y_t - t(x_t) %*% theta_hat)
  
  # Update parameter estimate theta_hat_t
  theta_hat <- theta_hat + solve(R) %*% x_t * err
  
  # Save results
  res <- rbind(res, data.frame(
    t = t,
    time = Dtrain$time[t],
    theta1 = theta_hat[1, 1],
    theta2 = theta_hat[2, 1]
  ))
}

# Print theta_hat for t = 1..3
print(res)

# Also print final theta_hat at t = 3
cat("\nFinal theta_hat at t = 3:\n")
print(theta_hat)
