% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/sdt.R
\name{sdt}
\alias{sdt}
\title{Generating data from the SDT model and computing the MSE}
\usage{
sdt(par, data, ntrials = 10000)
}
\arguments{
\item{par}{Numeric, should be a list of initial values for the response bias
and the internal noise.}

\item{data}{Dataframe, with observed prop_agree and prop_int1.}

\item{ntrials}{Numeric, number of trials per block (defaults to 10k).}
}
\value{
Numeric, the MSE.
}
\description{
Generating data from the SDT model and computing the MSE. Adapted from
Goupil et al. (2021), \url{https://doi.org/10.1038/s41467-020-20649-4}.
}
\examples{
\dontrun{
# example data (70\% agreement and no response bias)
df <- data.frame(prop_agree = 0.7, prop_int1 = 0.5, ntrials = 1e2)

# computing the MSE
sdt(par = c(0, 1), data = df)
}

}
\references{
Goupil, L., Ponsot, E., Richardson, D. et al. (2021). Listeners'
perceptions of the certainty and honesty of a speaker are associated with a
common prosodic signature. Nat Commun 12, 861. \url{https://doi.org/10.1038/s41467-020-20649-4}.
}
\author{
Ladislas Nalborczyk \email{ladislas.nalborczyk@gmail.com}.
}
