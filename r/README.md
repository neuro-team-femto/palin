
<!-- README.md is generated from README.Rmd. Please edit that file -->

# PALIN: Convert your PAs into INs

<!-- badges: start -->

<!-- badges: end -->

The goal of `palin` is to provide utilities for working with reverse
correlation data. It can be used, for instance, to estimate internal
noise (IN) from empirical percentages of agreement (PA) in such tasks.

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

### Computing kernels

Below we compute kernels using various methods.

``` r
library(patchwork)
library(tidyverse)
library(palin)

# importing some reverse correlation data (self-produced speech)
data(self_voice)
head(self_voice)
#>   participant block trial response    RT  feature     value
#> 1        01JM     1     1        0 0.298   0.0000  2.175316
#> 2        01JM     1     1        0 0.298 102.2548 -3.567088
#> 3        01JM     1     1        0 0.298 219.4468 -4.296887
#> 4        01JM     1     1        0 0.298 353.7580  1.499065
#> 5        01JM     1     1        0 0.298 507.6892  3.008010
#> 6        01JM     1     1        0 0.298 684.1064  1.302738

# computing the kernel via the difference method
diff_kernels <- computing_kernel(data = self_voice, method = "difference")
head(diff_kernels)
#>   participant  feature    negative    positive kernel_gain norm_kernel_gain
#> 1        01JM   0.0000 -0.38232369  0.14785399   0.5301777        0.6834186
#> 2        01JM 102.2548  0.01766484 -0.09270277  -0.1103676       -0.1422679
#> 3        01JM 219.4468  0.16504632  0.01236347  -0.1526829       -0.1968138
#> 4        01JM 353.7580  0.44920329 -0.67440862  -1.1236119       -1.4483773
#> 5        01JM 507.6892  0.71696975 -0.82346385  -1.5404336       -1.9856759
#> 6        01JM 684.1064  0.12811565  0.26302829   0.1349126        0.1739074

# computing the kernel using a GLM
glm_kernels <- computing_kernel(data = self_voice, method = "glm")
head(glm_kernels)
#>        feature       kernel        lower        upper
#> 1            0  0.006920131 -0.006040969  0.019892897
#> 2 102.25480495  0.011869046 -0.019442771  0.043196640
#> 3 219.44681723 -0.018932548 -0.050210661  0.012335792
#> 4  353.7580386 -0.019160194 -0.050435179  0.012104649
#> 5 507.68921389 -0.035676451 -0.067351520 -0.004027084
#> 6 684.10639247  0.006127961 -0.025210484  0.037477517

# plotting everything (see the doc via ?plot.kernel)
plot(diff_kernels, normalisation_method = "kernel_gain") +
    plot(glm_kernels, normalisation_method = "kernel_gain")
```

<img src="man/figures/README-kernels-1.png" width="100%" />

### Estimating internal noise

Computing response bias and internal noise from the percentage of
agreement.

``` r
# computing the percentage of agreement and the percentage of choosing the first
# stimulus in the double-pass trials
self_voice |>
    filter(participant == unique(self_voice$participant)[1]) |>
    response_consistency() |>
    select(participant, double_pass_prop_agree, double_pass_prop_first) |>
    distinct()
#>   participant double_pass_prop_agree double_pass_prop_first
#> 1        01JM                   0.66                   0.34

# estimating internal noise for this  participant
df <- data.frame(prop_agree = 0.66, prop_first = 0.34, ntrials = 100)

# fitting the SDT model to these data (using DEoptim and all available cores)
# the value represents the value of the cost function (log-MSE)
fit_results <- sdt_fitting(data = df, maxit = 100)
summary(fit_results)
#> 
#> ***** summary of DEoptim object ***** 
#> best member   :  0.66659 1.22712 
#> best value    :  -17.24203 
#> after         :  100 generations 
#> fn evaluated  :  2020 times 
#> *************************************
```

### Drift diffusion modelling

Fitting the drift diffusion model (DDM) to the responses and
distributions of RTs (after removing the last double-pass block).

``` r
library(fddm)

# reshaping the data
df <- self_voice |>
    # keeping only the first participant
    filter(participant == unique(self_voice$participant)[1]) |>
    # removing the last double-pass block
    filter(block < max(block) ) |>
    # keeping only the relevant columns
    select(participant, trial, response, RT) |>
    # removing duplicated rows
    distinct() |>
    # reshaping the resp variable as indicating int1 or int2
    mutate(
        resp = if_else(first(response) == 1, 0, 1),
        .by = c(participant, trial)
        ) |>
    distinct() |>
    # reshaping the resp variable
    mutate(resp = factor(ifelse(test = response == 0, yes = "lower", no = "upper") ) ) |>
    # removing extreme RTs
    filter(RT > 0.05 & RT < 2)

# plotting the RT distribution
# hist(df$RT, breaks = "FD")

# fitting the full DDM (pars are a, v, t0, w, sv)
# parameters are the threshold separation, drift rate, non-decision time,
# relative starting point, and inter-trial variability of drift rate
# verbose = 20 means that we want to print progress every 20 iterations
ddm_fit <- ddm_fitting(
    rt = df$RT,
    resp = df$resp,
    method = "DEoptim",
    maxit = 1e3
    )
summary(ddm_fit)
#> 
#> ***** summary of DEoptim object ***** 
#> best member   :  0 1.7089 0.01503 0.5 2.17958 
#> best value    :  585.5875 
#> after         :  1000 generations 
#> fn evaluated  :  50050 times 
#> *************************************

# comparing to the fit with ddm()
fddm_fit <- fddm::ddm(
    drift = RT + resp ~ 1,
    boundary = ~ 1,
    ndt = ~ 1,
    bias = ~ 1,
    sv = ~ 1,
    optim = "nlminb",
    args_optim = list(
        lo_bds = c(-5, 0, 0, 0, 0),
        up_bds = c(+5, 5, 5, 1, 5)
        ),
    data = df
    )
#> [1] "number of times we redid the initial parameters:"
#> [1] 0
#> t0[0] = 1.00296 >= 0.298
#> t0[116] = 0.105336 >= 0.103

# retrieving a summary
fddm_fit$coefficients
#> $drift
#>   (Intercept) 
#> -1.905569e-07 
#> 
#> $boundary
#> (Intercept) 
#>    1.708898 
#> 
#> $ndt
#> (Intercept) 
#>   0.0150254 
#> 
#> $bias
#> (Intercept) 
#>         0.5 
#> 
#> $sv
#> (Intercept) 
#>    2.179575
fddm_fit$loglik
#> [1] -585.5875
```

### Computing continous metrics of response consistency

``` r
# computing average consistency per participant and block
consistency <- response_consistency(
    data = self_voice,
    # method can be one of c("template_distance", "kernel_similarity", "intercept")
    method = "template_distance",
    double_pass = TRUE
    )

# plotting it
consistency %>%
    pivot_longer(cols = avg_consistency:weighted_consistency) %>%
    ggplot(aes(x = block, y = value, colour = name) ) +
    geom_line(
        aes(group = interaction(participant, name) ),
        linewidth = 0.5,
        alpha = 0.5,
        show.legend = FALSE
        ) +
    stat_summary(
        geom = "line",
        fun = median, linewidth = 1,
        show.legend = FALSE
        ) +
    facet_wrap(~name, ncol = 1) +
    theme_bw(base_size = 12, base_family = "Open Sans") +
    labs(x = "Block number", y = "Average consistency")
```

<img src="man/figures/README-consistency-1.png" width="100%" />
