% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/sdt.R
\name{sdt_fitting}
\alias{sdt_fitting}
\title{Fitting the SDT model}
\usage{
sdt_fitting(
  data,
  method = c("simulation", "expectation"),
  ntrials = 10000,
  log_mse = TRUE,
  fit_method = c("DEoptim", "nlminb", "SANN", "Nelder-Mead", "CG", "BFGS", "bobyqa",
    "grid"),
  cluster = NULL,
  grid_res = 0.01,
  maxit = 100,
  verbose = FALSE,
  return_grid = FALSE,
  smooth_grid = TRUE,
  smooth_k = 10
)
}
\arguments{
\item{data}{Dataframe, with observed prop_agree and prop_int1.}

\item{method}{Character, the method for computing prop_agree and prop_first ("simulation" or "expectation").}

\item{ntrials}{Numeric, number of simulated trials in the SDT model.}

\item{log_mse}{Logical, should we return the log-MSE (instead of the MSE).}

\item{fit_method}{Character, the optimisation method, see possible values below (DEoptim seems to work best).
Beware that method "grid" can take some time, depending on the size of the grid.}

\item{cluster}{Character, existing parallel cluster object. If provided, overrides + specified parallelType.}

\item{grid_res}{Numeric, grid resolution in units of response bias or internal
noise (only used for method "grid").}

\item{maxit}{Numeric, maximum number of iterations.}

\item{verbose}{Logical, whether to print progress during fitting.}

\item{return_grid}{Logical, should we return the full grid when method = "grid".}

\item{smooth_grid}{Logical, should we smooth the error surface (grid) with a GAM.}

\item{smooth_k}{Numeric, k value in mgcv::gam() for smoothing the error surface.}
}
\value{
The optimised parameter values and further convergence information.
}
\description{
Fitting the SDT model. NB: Best results are obtained with the "DEoptim"
method and at least 1e4 trials.
}
\examples{
\dontrun{
# generating prop_agree and prop_first from pars
sdt_df <- sdt_data(pars = c(1, 2), ntrials = 1e4)

# fitting the SDT model using bobyqa (fast method)
sdt_fitting(data = sdt_df, ntrials = 1e4, fit_method = "bobyqa")

fitting the SDT model using DEoptim (slower but more accurate)
sdt_fit <- sdt_fitting(data = sdt_df, ntrials = 1e4, fit_method = "DEoptim")
summary(sdt_fit)

fitting the SDT model using the grid method (super slower but accurate)
sdt_fitting(data = sdt_df, ntrials = 1e3, fit_method = "grid", grid_res = 0.1)
}

}
\author{
Ladislas Nalborczyk \email{ladislas.nalborczyk@gmail.com}.
}
