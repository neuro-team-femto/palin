#' Generating data from the SDT model
#'
#' Generating data from the SDT model adapted from
#' Goupil et al. (2021), \url{https://doi.org/10.1038/s41467-020-20649-4}.
#'
#' @param pars Numeric, should be a list of initial values for the response bias
#' and the internal noise.
#' @param return_summary Logical, should we return only prop_agree and prop_first, or the full data.
#' @param method Character, the method for computing prop_agree and prop_first ("simulation" or "expectation").
#' @param ntrials Numeric, number of trials per block (defaults to 1e4).
#'
#' @return Numeric, the MSE.
#'
#' @importFrom rlang .data
#'
#' @examples
#' \dontrun{
#' # generating prop_agree and prop_first from parameters values
#' sdt_data(pars = c(0, 1), ntrials = 1e4, method = "simulation")
#'
#' # generating prop_agree and prop_first from parameters values
#' sdt_data(pars = c(0, 1), ntrials = 1e4, method = "expectation")
#' }
#'
#' @author Ladislas Nalborczyk \email{ladislas.nalborczyk@@cnrs.fr}.
#'
#' @references Goupil, L., Ponsot, E., Richardson, D. et al. (2021). Listeners'
#' perceptions of the certainty and honesty of a speaker are associated with a
#' common prosodic signature. *Nature Communications 12*, 861. \url{https://doi.org/10.1038/s41467-020-20649-4}.
#'
#' @export
sdt_data <- function (pars, return_summary = TRUE, method = c("simulation", "expectation"), ntrials = 1e4) {

    # some tests for variable types
    stopifnot("ntrials must be a numeric..." = is.numeric(ntrials) )

    # ensuring that the method is one of the above
    method <- match.arg(method)

    # following Goupil et al. (2021)'s notation,
    # s_i is the difference between stim2 - stim1
    # sigma_ir is the difference between noise of stim2 and noise of stim1
    # prop_agree is the predicted percentage of agreement
    # prop_first is the predicted probability (percentage) of chosen first stimuli

    # response bias
    bias <- pars[[1]]

    # internal noise
    noise <- pars[[2]]

    if (method == "simulation") {

        # simulating data
        simulated_df <- data.frame(
            # difference in representation between stimuli (stim2 - stim1)
            s_i = rep(x = stats::rnorm(n = ntrials, mean = 0, sd = 1), 2),
            # numbering trials for each block
            trial = 1:ntrials,
            # identifying the two (repeated) blocks
            rep = rep(c("block1", "block2"), each = ntrials) ) |>
            # difference in noise added to the stimuli (sigma_stim2 - sigma_stim1)
            dplyr::mutate(
                sigma_ir = stats::rnorm(n = ntrials * 2, mean = 0, sd = noise)
                ) |>
            # decision rule
            dplyr::mutate(
                stim = dplyr::if_else(
                    condition = (.data$s_i + .data$sigma_ir) > bias,
                    # true = "stim1", false = "stim2"
                    true = "stim2", false = "stim1"
                    )
                ) |>
            # reshaping the dataframe
            tidyr::pivot_wider(
                names_from = .data$rep,
                values_from = .data$stim,
                # id_cols = .data$trial
                id_cols = c(.data$trial, .data$s_i)
                )

        if (return_summary == FALSE) {

            # returning the full data
            return (simulated_df)

        }

        # predicted probability (percentage) of choosing the first stimulus
        prop_first <- sum(simulated_df == "stim1") / (ntrials * 2)

        # predicted percentage of agreement
        prop_agree <- sum(simulated_df$block1 == simulated_df$block2) / ntrials

    } else if (method == "expectation") {

        # computing the probability of choosing the first interval
        prop_first <- 1 - stats::pnorm(bias / sqrt(1 + noise^2) )

        # computing the probability of agreement, which is a sum of
        # two conditional probabilities: p(both_stim1 | s_i) + p(both_stim2 | s_i)
        # involving the CDF of the standard normal distribution (since s_i ~ N(0, 1) )
        if (noise == 0) {

            # handling the hedge case of noise = 0
            prop_agree <- 1

        } else {

            # defining a function to compute agreement probability for a given s
            integrand <- function (s) {

                term <- stats::pnorm((s - bias) / noise)
                term_sq <- term^2 + (1 - term)^2

                # multiplying by N(0,1) density and returning the value
                return (term_sq * stats::dnorm(s) )

            }

            # integrating over all possible s
            prop_agree <- stats::integrate(integrand, lower = -Inf, upper = +Inf)$value

        }

    }

    # returning the predicted prop_agree and prop_first
    return (data.frame(prop_agree = prop_agree, prop_first = prop_first) )

}

#' Computing the MSE
#'
#' Computing the MSE from prop_agree and prop_first.
#'
#' @param pars Numeric, should be a list of initial values for the response bias
#' and the internal noise.
#' @param data Dataframe, with observed prop_agree and prop_first.
#' @param method Character, the method for computing prop_agree and prop_first ("simulation" or "expectation").
#' @param ntrials Numeric, number of trials per block (defaults to 1e4).
#' @param log_mse Logical, should we return the log-MSE (instead of the MSE).
#'
#' @examples
#' \dontrun{
#' # generating prop_agree and prop_first from pars
#' sdt_df <- sdt_data(pars = c(0, 1), ntrials = 1e4)
#'
#' # computing the MSE loss (or log-MSE)
#' sdt_loss(par = c(0, 1), data = sdt_df, ntrials = 1e4)
#' }
#'
#' @author Ladislas Nalborczyk \email{ladislas.nalborczyk@@cnrs.fr}.
#'
#' @export
sdt_loss <- function (pars, data, method = c("expectation", "simulation"), ntrials = 1e3, log_mse = TRUE) {

    # some tests for variable types
    stopifnot("data must be a dataframe..." = is.data.frame(data) )
    stopifnot("ntrials must be a numeric..." = is.numeric(ntrials) )

    # ensuring that the method is one of the above
    method <- match.arg(method)

    # simulating prop_agree and prop_first
    sdt_df <- sdt_data(pars = pars, method = method, ntrials = ntrials)

    # computing the squared prediction error (summed for the two percentages)
    prediction_error <- (data$prop_agree - sdt_df$prop_agree)^2 +
        (data$prop_first - sdt_df$prop_first)^2

    # returning the MSE (or log-MSE)
    if (log_mse == TRUE) {

        # log_prediction_error <- ifelse(
        #     test = prediction_error == 0,
        #     yes = log(1e-20),
        #     no = log(prediction_error)
        #     )
        return (log(prediction_error) )

    } else {

        return (prediction_error)

    }

}

#' Tabulate double-pass response patterns
#'
#' Converts full double-pass SDT data into counts for the four response patterns:
#' stim1_stim1, stim1_stim2, stim2_stim1, and stim2_stim2.
#'
#' @param data Data frame. Must contain columns `block1` and `block2`,
#'   as returned by `sdt_data(..., return_summary = FALSE)`.
#' @param response_levels Optional length-2 vector specifying the observed
#'   response labels corresponding to `stim1` and `stim2`. The first value is
#'   mapped to `stim1`, and the second value is mapped to `stim2`. For example,
#'   use `response_levels = c(0, 1)` when 0 means `stim1` and 1 means `stim2`.
#'
#' @return Named numeric vector of response-pattern counts.
#'
#' @export
sdt_multinom_counts <- function (data, response_levels = NULL) {

    stopifnot("data must be a dataframe..." = is.data.frame(data) )

    if (!all(c("block1", "block2") %in% names(data) ) ) {

        stop (
            "For multinomial fitting, data must contain block1 and block2. ",
            "Generate data with sdt_data(..., return_summary = FALSE), ",
            "or provide a data frame with block1 and block2 responses."
            )

    }

        # Convert to character internally so that numeric, factor, and character
    # response labels are handled consistently.
    block1 <- as.character(data$block1)
    block2 <- as.character(data$block2)

    if (any(is.na(block1)) || any(is.na(block2) ) ) {

        stop ("block1 and block2 must not contain missing values.")

    }

    observed_levels <- sort(unique(c(block1, block2) ) )

    if (is.null(response_levels) ) {

        # Default behaviour remains compatible with simulated palin data.
        if (setequal(observed_levels, c("stim1", "stim2") ) ) {

            response_levels <- c("stim1", "stim2")

        } else if (length(observed_levels) == 2) {

            response_levels <- observed_levels

            warning (
                "response_levels was not supplied. Inferring response_levels = c('",
                response_levels[1], "', '", response_levels[2], "'). ",
                "The first value is mapped to stim1 and the second to stim2. ",
                "If this is not intended, set response_levels explicitly."
            )

        } else {

            stop (
                "Could not infer response levels. ",
                "Please provide response_levels as a length-2 vector, e.g. c(0, 1)."
                )

        }

    } else {

        if (length(response_levels) != 2) {

            stop ("response_levels must be a length-2 vector.")

        }

        response_levels <- as.character(response_levels)

        unknown_levels <- setdiff(observed_levels, response_levels)

        if (length(unknown_levels) > 0) {

            stop (
                "Some response values are not present in response_levels: ",
                paste(unknown_levels, collapse = ", ")
                )

        }

    }

    # response_levels[1] is mapped to stim1
    # response_levels[2] is mapped to stim2
    recode_response <- function (x) {

        ifelse(
            x == response_levels[1],
            "stim1",
            ifelse(x == response_levels[2], "stim2", NA_character_)
            )

    }

    block1 <- recode_response(block1)
    block2 <- recode_response(block2)

    pattern <- paste(block1, block2, sep = "_")

    counts <- table(
        factor(
            pattern,
            levels = c(
                "stim1_stim1",
                "stim1_stim2",
                "stim2_stim1",
                "stim2_stim2"
                )
            )
        )

    counts <- as.numeric(counts)

    names(counts) <- c(
        "stim1_stim1",
        "stim1_stim2",
        "stim2_stim1",
        "stim2_stim2"
        )

    return (counts)

}

#' Predicted double-pass response-pattern probabilities
#'
#' Computes the predicted probabilities of the four double-pass response patterns
#' under the SDT model.
#'
#' @param pars Numeric vector of length 2. First value is response bias, second
#'   value is internal noise.
#' @param method Character. Either `"expectation"` or `"simulation"`.
#' @param ntrials Numeric. Number of trials used when `method = "simulation"`.
#' @param eps Numeric. Small positive value used to avoid probabilities of
#'   exactly zero.
#'
#' @return Named numeric vector of response-pattern probabilities.
#'
#' @export
sdt_multinom_probs <- function (
        pars,
        method = c("expectation", "simulation"),
        ntrials = 1e4,
        eps = 1e-12
        ) {

    method <- match.arg(method)

    bias <- pars[[1]]
    noise <- pars[[2]]

    if (method == "simulation") {

        sim_data <- sdt_data(
            pars = pars,
            method = "simulation",
            ntrials = ntrials,
            return_summary = FALSE
            )

        probs <- sdt_multinom_counts(sim_data)
        probs <- probs / sum(probs)

    } else if (method == "expectation") {

        if (noise < sqrt(.Machine$double.eps) ) {

            # Deterministic repeated choices when internal noise is 0.
            # Decision rule in sdt_data():
            # stim2 if s_i + sigma_ir > bias, otherwise stim1.
            p11 <- stats::pnorm(bias)
            p22 <- 1 - stats::pnorm(bias)
            p12 <- 0
            p21 <- 0

        } else {

            # Conditional probability of choosing stim2 for a given latent s.
            p_stim2_given_s <- function (s) {

                stats::pnorm((s - bias) / noise)

            }

            integrate_over_s <- function (fun) {
                stats::integrate(
                    f = function (s) fun(s) * stats::dnorm(s),
                    lower = -Inf,
                    upper = +Inf,
                    subdivisions = 200L,
                    rel.tol = .Machine$double.eps^0.25
                    )$value
            }

            # P(stim1 in block1, stim1 in block2)
            p11 <- integrate_over_s(
                function (s) (1 - p_stim2_given_s(s) )^2
                )

            # P(stim2 in block1, stim2 in block2)
            p22 <- integrate_over_s(
                function (s) p_stim2_given_s(s)^2
                )

            # Off-diagonal cells are symmetric under the current model.
            p12 <- (1 - p11 - p22) / 2
            p21 <- p12

        }

        probs <- c(
            stim1_stim1 = p11,
            stim1_stim2 = p12,
            stim2_stim1 = p21,
            stim2_stim2 = p22
            )

    }

    # Avoid log(0). Renormalise after clipping.
    probs <- pmax(probs, eps)
    probs <- probs / sum(probs)

    return (probs)

}

#' Multinomial negative log-likelihood for the SDT model
#'
#' Computes the negative multinomial log-likelihood of the observed double-pass
#' response patterns under the SDT model.
#'
#' @param pars Numeric vector of length 2. First value is response bias, second
#'   value is internal noise.
#' @param data Data frame. Must contain columns `block1` and `block2`,
#'   as returned by `sdt_data(..., return_summary = FALSE)`.
#' @param method Character. Either `"expectation"` or `"simulation"`.
#' @param ntrials Numeric. Number of trials used when `method = "simulation"`.
#' @param eps Numeric. Small positive value used to avoid probabilities of
#'   exactly zero.
#' @param response_levels Optional length-2 vector specifying the observed
#'   response labels corresponding to `stim1` and `stim2`. The first value is
#'   mapped to `stim1`, and the second value is mapped to `stim2`. For example,
#'   use `response_levels = c(0, 1)` when 0 means `stim1` and 1 means `stim2`.
#' @param ... Additional arguments. Currently unused, but accepted for
#'   compatibility with optimisation functions.
#'
#' @return Numeric value: the negative multinomial log-likelihood.
#'
#' @export
sdt_multinom_loss <- function(
        pars,
        data,
        method = c("expectation", "simulation"),
        ntrials = 1e4,
        eps = 1e-12,
        response_levels = NULL,
        ...
        ) {

    method <- match.arg(method)

    counts <- sdt_multinom_counts(
        data = data,
        response_levels = response_levels
        )

    probs <- sdt_multinom_probs(
        pars = pars,
        method = method,
        ntrials = ntrials,
        eps = eps
        )

    # Negative multinomial log-likelihood, dropping the multinomial constant.
    # The constant does not affect optimisation.
    nll <- -sum(counts * log(probs) )

    return (nll)

}

#' Fitting the SDT model
#'
#' Fitting the SDT model. NB: Best results are obtained with the "DEoptim"
#' method and at least 1e4 trials.
#'
#' @param data Dataframe, with observed prop_agree and prop_int1.
#' @param method Character, the method for computing prop_agree and prop_first ("simulation" or "expectation").
#' @param ntrials Numeric, number of simulated trials in the SDT model.
#' @param log_mse Logical, should we return the log-MSE (instead of the MSE).
#' @param loss Character. Loss function to minimise. Either `"mse"` for the
#'   original mean squared error on summary proportions, or `"multinomial"` for
#'   the multinomial negative log-likelihood on full double-pass response
#'   patterns.
#' @param fit_method Character, the optimisation method, see possible values below (DEoptim seems to work best).
#' Beware that method "grid" can take some time, depending on the size of the grid.
#' @param cluster Character, existing parallel cluster object. If provided, overrides + specified parallelType.
#' @param grid_res Numeric, grid resolution in units of response bias or internal
#' noise (only used for method "grid").
#' @param maxit Numeric, maximum number of iterations.
#' @param internal_noise_upper_bound Numeric, upper bound for internal noise estimation.
#' @param confint Logical, whether to compute profile-likelihood confidence
#'   intervals for response bias and internal noise. Only available when
#'   `loss = "multinomial"`.
#' @param conf_level Numeric, confidence level for profile-likelihood confidence
#'   intervals.
#' @param confint_grid_length Numeric, number of grid points used for each
#'   one-dimensional profile likelihood.
#' @param return_profiles Logical, whether to return the full profile-likelihood
#'   objects in addition to the confidence interval table.
#' @param verbose Logical, whether to print progress during fitting.
#' @param return_grid Logical, should we return the full grid when method = "grid".
#' @param smooth_grid Logical, should we smooth the error surface (grid) with a GAM.
#' @param plot_surface Logical, should we plot the GAM-smoothed error surface.
#' @param fine_res Numeric, finer grid resolution for plotting the GAM-smoothed error surface.
#' @param smooth_k Numeric, k value in mgcv::gam() for smoothing the error surface.
#' @param response_levels Optional length-2 vector specifying the observed
#'   response labels corresponding to `stim1` and `stim2`. The first value is
#'   mapped to `stim1`, and the second value is mapped to `stim2`. For example,
#'   use `response_levels = c(0, 1)` when 0 means `stim1` and 1 means `stim2`.
#'
#' @return The optimised parameter values and further convergence information.
#'   If `confint = TRUE`, the returned object also contains profile-likelihood
#'   confidence intervals.
#'
#' @importFrom stats nlminb optim
#'
#' @examples
#' \dontrun{
#' # Generate full double-pass data for multinomial fitting
#' sdt_df <- sdt_data(
#'     pars = c(1, 2),
#'     method = "simulation",
#'     ntrials = 200,
#'     return_summary = FALSE
#'     )
#'
#' # Fit with the multinomial likelihood using DEoptim
#' sdt_fit <- sdt_fitting(
#'     data = sdt_df,
#'     method = "expectation",
#'     loss = "multinomial",
#'     fit_method = "DEoptim",
#'     ntrials = nrow(sdt_df),
#'     maxit = 200,
#'     internal_noise_upper_bound = 10,
#'     confint = TRUE
#'     )
#'
#' # Point estimates and profile-likelihood confidence intervals
#' sdt_fit$confint
#'
#' # MSE fitting on summary proportions
#' sdt_summary <- sdt_data(pars = c(1, 2), ntrials = 200)
#'
#' sdt_summary_fit <- sdt_fitting(
#'     data = sdt_summary,
#'     method = "expectation",
#'     loss = "mse",
#'     fit_method = "DEoptim",
#'     ntrials = 1000,
#'     maxit = 200
#'     )
#'
#' # Point estimates
#' sdt_summary_fit$optim$bestmem
#' }
#' @author Ladislas Nalborczyk \email{ladislas.nalborczyk@@cnrs.fr}.
#'
#' @export
sdt_fitting <- function (
        data,
        method = c("expectation", "simulation"),
        ntrials = 1e4,
        log_mse = TRUE,
        loss = c("multinomial", "mse"),
        fit_method = c("DEoptim", "nlminb", "SANN", "Nelder-Mead", "CG", "BFGS", "bobyqa", "grid"),
        cluster = NULL,
        grid_res = 0.05,
        maxit = 100,
        internal_noise_upper_bound = 10,
        confint = TRUE,
        conf_level = 0.95,
        confint_grid_length = 501,
        return_profiles = FALSE,
        verbose = FALSE,
        return_grid = FALSE,
        smooth_grid = TRUE,
        plot_surface = TRUE,
        fine_res = 200,
        smooth_k = 20,
        response_levels = NULL
        ) {

    # some tests for variable types
    stopifnot("data must be a dataframe..." = is.data.frame(data) )
    stopifnot("ntrials must be a numeric..." = is.numeric(ntrials) )
    stopifnot("grid_res must be a numeric..." = is.numeric(grid_res) )
    stopifnot("confint must be logical..." = is.logical(confint) )
    stopifnot("conf_level must be numeric..." = is.numeric(conf_level) )
    stopifnot("confint_grid_length must be numeric..." = is.numeric(confint_grid_length) )
    stopifnot("return_profiles must be logical..." = is.logical(return_profiles) )

    # method should be one of above
    method <- match.arg(method)

    # fit_method should be one of above
    fit_method <- match.arg(fit_method)

    loss <- match.arg(loss)

    if (isTRUE(confint) && loss != "multinomial") {

        warning(
            "confint = TRUE is only available when loss = 'multinomial'. ",
            "Setting confint = FALSE."
            )

        confint <- FALSE

    }

    # loss_fun <- switch (
    #     loss,
    #     mse = sdt_loss,
    #     multinomial = sdt_multinom_loss
    #     )

    loss_fun <- switch (
        loss,

        mse = function (pars, data, method, ntrials, log_mse, ...) {

            sdt_loss(
                pars = pars,
                data = data,
                method = method,
                ntrials = ntrials,
                log_mse = log_mse
                )

        },

        multinomial = function (pars, data, method, ntrials, log_mse, ...) {

            sdt_multinom_loss(
                pars = pars,
                data = data,
                method = method,
                ntrials = ntrials,
                response_levels = response_levels
                )

        }

    )

    # helper functions
    extract_sdt_pars <- function (fit, fit_method) {

        if (fit_method == "DEoptim") {

            pars_hat <- as.numeric(fit$optim$bestmem)

        } else if (fit_method == "nlminb") {

            pars_hat <- as.numeric(fit$par)

        } else if (fit_method == "SANN") {

            pars_hat <- as.numeric(fit$par)

        } else if (fit_method %in% c("Nelder-Mead", "CG", "BFGS", "bobyqa") ) {

            if (all(c("bias", "noise") %in% names(fit) ) ) {

                pars_hat <- as.numeric(fit[1, c("bias", "noise")])

            } else {

                pars_hat <- as.numeric(fit[1, 1:2])

            }

        } else if (fit_method == "grid") {

            if (all(c("response_bias", "internal_noise") %in% names(fit) ) ) {

                pars_hat <- c(fit$response_bias[1], fit$internal_noise[1])

            } else if (all(c("best_bias", "best_noise") %in% names(fit) ) ) {

                pars_hat <- c(fit$best_bias[1], fit$best_noise[1])

            } else {

                stop (
                    "Cannot extract point estimates from grid output. ",
                    "Use return_grid = FALSE when confint = TRUE."
                    )

            }

        }

        names(pars_hat) <- c("bias", "internal_noise")
        pars_hat

    }

    add_sdt_confint <- function (fit) {

        if (loss != "multinomial") {

            stop ("Profile confidence intervals are only available for loss = 'multinomial'.")

        }

        if (!all(c("block1", "block2") %in% names(data) ) ) {

            stop (
                "Profile confidence intervals require full double-pass data. ",
                "Generate data with sdt_data(..., return_summary = FALSE)."
                )

        }

        pars_hat <- extract_sdt_pars(fit, fit_method)

        ci_res <- sdt_profile_confint(
            data = data,
            pars_hat = pars_hat,
            method = method,
            ntrials = ntrials,
            bias_bounds = c(-5, 5),
            noise_bounds = c(0, internal_noise_upper_bound),
            grid_length = confint_grid_length,
            level = conf_level
            )

        if (is.data.frame(fit) && nrow(fit) == 1) {

            fit$bias_lower <- ci_res$ci$lower[ci_res$ci$parameter == "bias"]
            fit$bias_upper <- ci_res$ci$upper[ci_res$ci$parameter == "bias"]
            fit$internal_noise_lower <- ci_res$ci$lower[
                ci_res$ci$parameter == "internal_noise"
                ]
            fit$internal_noise_upper <- ci_res$ci$upper[
                ci_res$ci$parameter == "internal_noise"
                ]

            attr(fit, "confint") <- ci_res$ci

            if (isTRUE(return_profiles) ) {

                attr(fit, "profiles") <- ci_res$profiles

            }

        } else {

            fit$confint <- ci_res$ci

            if (isTRUE(return_profiles) )
                {
                fit$profiles <- ci_res$profiles

            }

        }

        return (fit)

    }

    if (fit_method == "DEoptim") {

        # starting the optimisation
        fit <- DEoptim::DEoptim(
            fn = loss_fun,
            data = data,
            method = method,
            ntrials = ntrials,
            log_mse = log_mse,
            lower = c(-5, 0),
            upper = c(+5, internal_noise_upper_bound),
            control = DEoptim::DEoptim.control(
                # maximum number of iterations
                itermax = maxit,
                # printing progress iteration
                trace = verbose,
                # defines the differential evolution strategy (defaults to 2)
                # 1: DE / rand / 1 / bin (classical strategy)
                # 2: DE / local-to-best / 1 / bin (default)
                # 3: DE / best / 1 / bin with jitter
                # 4: DE / rand / 1 / bin with per-vector-dither
                # 5: DE / rand / 1 / bin with per-generation-dither
                # 6: DE / current-to-p-best / 1
                # strategy = 3,
                # value to reach (defaults to -Inf)
                # VTR = 0,
                # number of population members (by default 10*length(lower) )
                # NP = 200,
                # NP = nrow(lhs_initial_pop),
                # F is the mutation constant (defaults to 0.8)
                # F = 0.9,
                # crossover probability (recombination) (defaults to 0.5)
                # CR = 0.9,
                # c controls the speed of the crossover adaptation
                # when strategy = 6 (defaults to 0)
                # c = 0.1,
                # proportion of best solutions to use in the mutation
                # when strategy = 6 (defaults to 0.2)
                # p = 0.1,
                # defining the initial population using lhs
                # initialpop = lhs_initial_pop,
                # when to stop optimisation
                reltol = 1e-9,
                # number of iteration after which to stop the optimisation
                # if there is no improvement
                # steptol = 1000,
                # using all available cores
                parallelType = "parallel",
                # defining the package to be imported on each parallel core
                packages = c("DEoptim", "dplyr", "tidyr", "palin"),
                # defining the cluster
                cluster = cluster
                )
            )

    } else if (fit_method == "nlminb") {

        fit <- stats::nlminb(
            start = c(0, 1),
            objective = loss_fun,
            data = data,
            method = method,
            ntrials = ntrials,
            log_mse = log_mse,
            lower = c(-5, 0),
            upper = c(+5, internal_noise_upper_bound)
            )

    } else if (fit_method == "SANN") {

        fit <- stats::optim(
            par = c(bias = 0, noise = 1),
            fn = loss_fun,
            data = data,
            ntrials = ntrials,
            log_mse = log_mse,
            method = fit_method
            )

    } else if (fit_method %in% c("Nelder-Mead", "CG", "BFGS", "bobyqa") ) {

        fit <- optimx::optimx(
            par = c(bias = 0, noise = 1),
            fn = loss_fun,
            data = data,
            ntrials = ntrials,
            log_mse = log_mse,
            method = fit_method
            )

    } else if (fit_method == "grid") {

        # starting parallel mode (by default using all available cores except one)
        future::plan(future::multisession(workers = parallel::detectCores() - 1) )

        # grid search: computing the MSE for many possible values of
        # response bias and internal noise (may take some time...)
        param_grid <- tidyr::crossing(
            x = seq(from = -5, to = +5, by = grid_res),
            y = seq(from = 0, to = internal_noise_upper_bound, by = grid_res)
            )

        # warning the user about the number of simulation to evaluate...
        message(
            paste(
                "palin will now explore",
                nrow(param_grid),
                "combinations of parameters values, so please adjust your expectations accordingly..."
                )
            )

        # setting up the progress bar (cf. https://progressr.futureverse.org)
        progressr::handlers(global = TRUE)

        # initialising the progress bar
        p <- progressr::progressor(steps = nrow(param_grid) )

        # computing the error for many possible parameters values
        param_grid <- dplyr::mutate(
            .data = param_grid,
            z = future.apply::future_apply(
                X = param_grid,
                MARGIN = 1,
                FUN = function (x, ...) {
                    p(sprintf("x=%g", x) )
                    # palin::sdt_loss(x, data = data, ntrials = ntrials, log_mse = log_mse)
                    loss_fun(
                        pars = x,
                        data = data,
                        method = method,
                        ntrials = ntrials,
                        log_mse = log_mse
                        )
                    },
                future.seed = NULL
                )
            )

        # explicitly closing multisession workers by switching plan back to sequential
        future::plan(future::sequential)

        # finding the minimum (or minima) bias and noise values
        minima <- which(param_grid$z == min(param_grid$z) )

        # or looking for the minimum on a smoothed grid/surface
        if (smooth_grid == TRUE) {

            message ("Fitting a GAM to smooth the error function...")

            # smoothing the error surface with a GAM
            smoothing_model <- mgcv::gam(
                formula = z ~ te(x, y, k = smooth_k),
                data = param_grid
                )

            # making predictions about z
            param_grid$z_smoothed <- stats::fitted(smoothing_model)

            # finds the minimum (or minima) bias and noise values
            # minima <- which(param_grid$z_smoothed == min(param_grid$z_smoothed) )

            # finds the minimum of the fitted surface via optimisation
            gam_objective <- function (par, model) {

                newdata <- data.frame(
                    x = par[1],
                    y = par[2]
                    )
                pred <- stats::predict(model, newdata = newdata)
                as.numeric(pred)

            }

            smoothed_grid_min_idx <- which(
                param_grid$z_smoothed == min(param_grid$z_smoothed, na.rm = TRUE)
                )

            if (length(smoothed_grid_min_idx) == 1) {

                start_x <- param_grid$x[smoothed_grid_min_idx]
                start_y <- param_grid$y[smoothed_grid_min_idx]

            } else {

                start_x <- stats::median(param_grid$x[smoothed_grid_min_idx])
                start_y <- stats::median(param_grid$y[smoothed_grid_min_idx])

            }

            opt_res <- stats::optim(
                par = c(start_x, start_y),
                fn = gam_objective,
                model = smoothing_model,
                method = "L-BFGS-B",
                lower = c(min(param_grid$x), min(param_grid$y) ),
                upper = c(max(param_grid$x), max(param_grid$y) )
                )

            best_bias <- opt_res$par[1]
            best_noise <- opt_res$par[2]
            best_mse <- opt_res$value

            gam_minimum <- data.frame(
                best_bias = best_bias,
                best_noise = best_noise,
                best_mse = best_mse
                )

            if (isTRUE(plot_surface) ) {

                vis_grid <- tidyr::crossing(
                    x = seq(
                        min(param_grid$x),
                        max(param_grid$x),
                        length.out = fine_res
                        ),
                    y = seq(
                        min(param_grid$y),
                        max(param_grid$y),
                        length.out = fine_res
                        )
                    )

                vis_grid$z_pred <- as.numeric(
                    stats::predict(
                        smoothing_model,
                        newdata = vis_grid
                        )
                    )

                p_surface <- ggplot2::ggplot() +
                    ggplot2::geom_raster(
                        data = vis_grid,
                        ggplot2::aes(x = .data$x, y = .data$y, fill = .data$z_pred),
                        interpolate = TRUE,
                        show.legend = FALSE
                        ) +
                    ggplot2::geom_contour(
                        data = vis_grid,
                        ggplot2::aes(x = .data$x, y = .data$y, z = .data$z_pred),
                        colour = "white",
                        linewidth = 0.25
                        ) +
                    ggplot2::geom_point(
                        data = gam_minimum,
                        ggplot2::aes(x = .data$best_bias, y = .data$best_noise),
                        shape = 4,
                        size = 4,
                        stroke = 1.4,
                        colour = "red"
                        ) +
                    ggplot2::labs(
                        title = "GAM-smoothed error surface",
                        subtitle = paste0(
                            "Estimated minimum at bias = ",
                            round(best_bias, 3),
                            ", noise = ",
                            round(best_noise, 3),
                            ", MSE = ",
                            round(best_mse, 5)
                            ),
                        x = "Response bias",
                        y = "Internal noise",
                        fill = "GAM-predicted loss"
                        ) +
                    ggplot2::scale_fill_viridis_c(option = "magma") +
                    ggplot2::theme_bw(base_size = 12, base_family = "Open Sans") +
                    ggplot2::scale_x_continuous(expand = c(0, 0) ) +
                    ggplot2::scale_y_continuous(expand = c(0, 0) )

                print(p_surface)

            }

            # return (gam_minimum)
            fit <- gam_minimum

            if (isTRUE(confint) ) {

                fit <- add_sdt_confint(fit)

            }

            return (fit)

        }

        if (length(minima) == 1) {

            # if there is only one minimum, returns it
            avg_bias <- param_grid$x[minima]
            avg_noise <- param_grid$y[minima]
            fit <- data.frame(response_bias = avg_bias, internal_noise = avg_noise)

            } else {

                # otherwise, finds the average (median) bias and noise values across the minima
                message ("Several minima found, returning the median parameter values...")
                avg_bias <- stats::median(param_grid$x[minima])
                avg_noise <- stats::median(param_grid$y[minima])
                fit <- data.frame(response_bias = avg_bias, internal_noise = avg_noise)

            }

        # should we return the full grid?
        if (return_grid) fit <- param_grid

    }

    if (isTRUE(confint) ) {

        fit <- add_sdt_confint(fit)

    }

    # returning the fit
    return (fit)

}
