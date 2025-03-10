% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/sdt.R
\name{sdt_fitting}
\alias{sdt_fitting}
\title{Fitting the SDT model}
\usage{
sdt_fitting(
  data,
  ntrials = 10000,
  method = c("DEoptim", "nlminb", "SANN", "Nelder-Mead", "CG", "BFGS", "bobyqa", "grid"),
  cluster = NULL,
  grid_res = 0.01,
  maxit = 100,
  verbose = FALSE
)
}
\arguments{
\item{data}{Dataframe, with observed prop_agree and prop_int1.}

\item{ntrials}{Numeric, number of simulated trials in the SDT model.}

\item{method}{Character, the optimisation method, see possible values below (DEoptim seems to work best).
Beware that method "grid" can take some time, depending on the size of the grid.}

\item{cluster}{Character, existing parallel cluster object. If provided, overrides + specified parallelType.}

\item{grid_res}{Numeric, grid resolution in units of response bias or internal
noise (only used for method "grid").}

\item{maxit}{Numeric, maximum number of iterations.}

\item{verbose}{Boolean, whether to print progress during fitting.}
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
# using summary statistics of empirical data
df <- data.frame(prop_agree = 0.7, prop_int1 = 0.413, ntrials = 1e2)

# fitting the SDT model using bobyqa (fast method)
sdt_fitting(data = df, ntrials = 1e4, method = "bobyqa")

fitting the SDT model using DEoptim (slower but more accurate)
sdt_fit <- sdt_fitting(data = df, ntrials = 1e4, method = "DEoptim")
summary(sdt_fit)

fitting the SDT model using the grid method (super slower but more accurate)
sdt_fitting(data = df, ntrials = 1e3, method = "grid", grid_res = 0.05)
}

}
\author{
Ladislas Nalborczyk \email{ladislas.nalborczyk@gmail.com}.
}
