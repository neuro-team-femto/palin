#' Profile-likelihood confidence intervals for SDT parameters
#'
#' Computes approximate profile-likelihood confidence intervals for response bias
#' and internal noise using the multinomial negative log-likelihood.
#'
#' @param data Data frame with columns `block1` and `block2`, as returned by
#'   `sdt_data(..., return_summary = FALSE)`.
#' @param pars_hat Optional numeric vector of length 2. Estimated bias and
#'   internal noise. If `NULL`, the function estimates them first.
#' @param method Character. Usually `"expectation"`.
#' @param ntrials Numeric. Number of trials, passed to `sdt_multinom_loss()`.
#' @param bias_bounds Numeric vector of length 2.
#' @param noise_bounds Numeric vector of length 2.
#' @param grid_length Numeric. Number of profile points per parameter.
#' @param level Numeric. Confidence level.
#'
#' @return A list containing point estimates, confidence intervals, and profiles.
#'
#' @noRd
sdt_profile_confint <- function (
        data,
        pars_hat = NULL,
        method = "expectation",
        ntrials = nrow(data),
        bias_bounds = c(-5, 5),
        noise_bounds = c(0, 10),
        grid_length = 501,
        level = 0.95
        ) {

    stopifnot("data must be a dataframe..." = is.data.frame(data) )

    bounds_lower <- c(bias_bounds[1], noise_bounds[1])
    bounds_upper <- c(bias_bounds[2], noise_bounds[2])

    nll <- function (pars) {

        sdt_multinom_loss(
            pars = pars,
            data = data,
            method = method,
            ntrials = ntrials
            )

    }

    if (is.null(pars_hat) ) {

        fit <- stats::optim(
            par = c(0, 1),
            fn = nll,
            method = "L-BFGS-B",
            lower = bounds_lower,
            upper = bounds_upper
            )

        pars_hat <- fit$par
        nll_min <- fit$value

    } else {

        nll_min <- nll(pars_hat)

    }

    cutoff <- stats::qchisq(level, df = 1) / 2

    profile_one <- function (param_id) {

        grid <- if (param_id == 1) {

            seq(bias_bounds[1], bias_bounds[2], length.out = grid_length)

        } else {

            seq(noise_bounds[1], noise_bounds[2], length.out = grid_length)

        }

        prof <- lapply(grid, function (value) {

            if (param_id == 1) {

                opt <- stats::optimize(
                    f = function (noise) nll(c(value, noise) ),
                    interval = noise_bounds
                    )

                out <- data.frame(
                    fixed_value = value,
                    nuisance_value = opt$minimum,
                    nll = opt$objective
                    )

            } else {

                opt <- stats::optimize(
                    f = function (bias) nll(c(bias, value) ),
                    interval = bias_bounds
                    )

                out <- data.frame(
                    fixed_value = value,
                    nuisance_value = opt$minimum,
                    nll = opt$objective
                    )

            }

            out

        })

        prof <- do.call(rbind, prof)
        prof$delta_nll <- prof$nll - nll_min
        prof

    }

    bias_profile <- profile_one(param_id = 1)
    noise_profile <- profile_one(param_id = 2)

    extract_ci <- function (profile) {

        keep <- profile$delta_nll <= cutoff

        if (!any(keep) ) {

            return (c(lower = NA_real_, upper = NA_real_) )

        }

        c(
            lower = min(profile$fixed_value[keep]),
            upper = max(profile$fixed_value[keep])
            )

    }

    ci_bias <- extract_ci(bias_profile)
    ci_noise <- extract_ci(noise_profile)

    ci <- data.frame(
        parameter = c("bias", "internal_noise"),
        estimate = c(pars_hat[1], pars_hat[2]),
        lower = c(ci_bias["lower"], ci_noise["lower"]),
        upper = c(ci_bias["upper"], ci_noise["upper"]),
        level = level,
        hits_lower_bound = c(
            ci_bias["lower"] <= bias_bounds[1],
            ci_noise["lower"] <= noise_bounds[1]
            ),
        hits_upper_bound = c(
            ci_bias["upper"] >= bias_bounds[2],
            ci_noise["upper"] >= noise_bounds[2]
            )
    )

    rownames(ci) <- NULL

    return (
        list(
            estimates = c(bias = pars_hat[1], internal_noise = pars_hat[2]),
            nll_min = nll_min,
            cutoff = cutoff,
            ci = ci,
            profiles = list(
                bias = bias_profile,
                internal_noise = noise_profile
                )
            )
        )

}
