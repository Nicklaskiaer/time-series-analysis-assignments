myKalmanFilter <- function(
  y,              # Vector of observations y_t
  theta,          # Model parameters: theta = c(a, b, sigma1)
  R,              # Measurement noise variance
  x_prior = 0,    # Initial prior mean for X_0
  P_prior = 10    # Initial prior variance for X_0
) {
  
  # Extract parameters
  a <- theta[1]
  b <- theta[2]
  sigma1 <- theta[3]
  
  # Number of observations
  N <- length(y)
  
  # Storage vectors
  x_pred  <- numeric(N)  # Predicted state means
  P_pred  <- numeric(N)  # Predicted state variances
  x_filt  <- numeric(N)  # Filtered state means
  P_filt  <- numeric(N)  # Filtered state variances
  
  innovation     <- numeric(N)  # Prediction errors: y[t] - x_pred[t]
  innovation_var <- numeric(N)  # Variance of prediction errors
  
  for (t in seq_len(N)) {
    
    # -----------------------------
    # Prediction step
    # -----------------------------
    
    if (t == 1) {
      # For the first observation, use the initial prior directly
      x_pred[t] <- x_prior
      P_pred[t] <- P_prior
    } else {
      # Predict the current state using the previous filtered estimate
      x_pred[t] <- a * x_filt[t - 1] + b
      
      # Predict the current variance
      # Variance is propagated through the model and system noise is added
      P_pred[t] <- a^2 * P_filt[t - 1] + sigma1^2
    }
    
    # -----------------------------
    # Update step
    # -----------------------------
    
    # Innovation: difference between observation and prediction
    innovation[t] <- y[t] - x_pred[t]
    
    # Innovation variance: uncertainty from prediction plus measurement noise
    innovation_var[t] <- P_pred[t] + R
    
    # Kalman gain: determines how much we trust the observation
    K_t <- P_pred[t] / innovation_var[t]
    
    # Update state estimate using the observation
    x_filt[t] <- x_pred[t] + K_t * innovation[t]
    
    # Update state uncertainty
    P_filt[t] <- (1 - K_t) * P_pred[t]
  }
  
  return(list(
    x_pred = x_pred,
    P_pred = P_pred,
    x_filt = x_filt,
    P_filt = P_filt,
    innovation = innovation,
    innovation_var = innovation_var
  ))
}