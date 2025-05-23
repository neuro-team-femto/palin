% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/consistency.R
\name{response_consistency}
\alias{response_consistency}
\title{Computing different metrics of response's consistency.}
\usage{
response_consistency(
  data,
  participant_id = "participant",
  block_id = "block",
  trial_id = "trial",
  feature_id = "feature",
  value_id = "value",
  response_id = "response",
  method = c("template_distance", "kernel_similarity", "intercept"),
  double_pass = TRUE
)
}
\arguments{
\item{data}{Dataframe, with reverse correlation data (in long format).}

\item{participant_id}{Numeric/Character/Factor, column in data specifying the participant ID.}

\item{block_id}{Numeric, column in data specifying the block ID.}

\item{trial_id}{Numeric, column in data specifying the trial ID.}

\item{feature_id}{Numeric/Factor, column in data specifying the feature.}

\item{value_id}{Numeric, column in data specifying the feature value}

\item{response_id}{Numeric, column in data specifying the response.}

\item{method}{Character, which consistency method to use.}

\item{double_pass}{Logical, indicating whether the last block was repeated.}
}
\value{
Dataframe, various metrics of response consistency.
}
\description{
Computing different metrics of response's consistency from both double-pass
and non double-pass reverse correlation data.
}
\examples{
\dontrun{
# importing the self-voice data
data(self_voice)
head(self_voice)

# computing metrics of response consistency per participant and per block
response_consistency(self_voice, method = "template_distance") |> head(10)
response_consistency(self_voice, method = "kernel_similarity") |> head(10)
response_consistency(self_voice, method = "intercept") |> head(10)
}

}
\author{
Ladislas Nalborczyk \email{ladislas.nalborczyk@gmail.com}.
}
