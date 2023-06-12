
<!-- README.md is generated from README.Rmd. Please edit that file -->

# PALIN: Convert your PAs into INs

<!-- badges: start -->
<!-- badges: end -->

The goal of `palin` is to provide utilities for working with reverse
correlation data. It can be used, for instance, to estimate internal
noise in such tasks.

## Installation

You can install the development version of `palin` from GitHub with:

``` r
install.packages("remotes")

remotes::install_github(
    repo = "https://github.com/neuro-team-femto/palin/r",
    dependencies = TRUE
    )
```

## Usage

Computing response bias and internal noise from the percentage of
agreement.

``` r
library(palin)

# example data (70% agreement and no response bias)
df <- data.frame(prop_agree = 0.7, prop_int1 = 0.5, ntrials = 1e2)

# fitting the SDT model to these data
sdt_fitting(data = df, ntrials = 1e4, method = "bobyqa")
#>              bias     noise        value fevals gevals niter convcode  kkt1
#> bobyqa 0.01510304 0.9101031 0.0005363325     30     NA    NA        0 FALSE
#>        kkt2 xtime
#> bobyqa TRUE 0.963
```

Fitting the drift diffusion model to the raw data (minus the last
block).

``` r
library(tidyverse)

# importing the data
data(self_produced_speech)
head(self_produced_speech)
#>   participant sex age block trial response    RT
#> 1          LN   M  30     1     1    stim2 0.267
#> 2          LN   M  30     1     2    stim2 0.274
#> 3          LN   M  30     1     3    stim2 0.128
#> 4          LN   M  30     1     4    stim2 0.359
#> 5          LN   M  30     1     5    stim2 0.233
#> 6          LN   M  30     1     6    stim2 0.284

# reshaping the data
df <- self_produced_speech |>
    # removing the last block
    filter(block < 7) |>
    # keeping only the relevant columns
    select(participant, choice = response, RT) |>
    mutate(choice = ifelse(test = choice == "stim1", yes = 1, no = 2) )

# splitting data by participant
df_ln <- df |> filter(participant == "LN")
df_mm <- df |> filter(participant == "MM")

# fitting the full DDM for LN and MM (pars are a, v, t0, w, sv)
ddm_fitting(rt = df_ln$RT, resp = df_ln$choice, method = "nlminb")$par
#> [1] 0.9621730 0.4611907 0.0000000 0.4841055 0.8369784
ddm_fitting(rt = df_mm$RT, resp = df_mm$choice, method = "nlminb")$par
#> [1]  1.6224599 -0.2767494  0.1767362  0.5363406  0.8699803
```
