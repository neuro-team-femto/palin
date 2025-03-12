#' Generating data from the SDT model
#'
#' Generating data from the SDT model and computing the MSE. Adapted from
#' Goupil et al. (2021), \url{https://doi.org/10.1038/s41467-020-20649-4}.
#'
#' @param pars Numeric, should be a list of initial values for the response bias
#' and the internal noise.
#' @param method Character, the method for computing prop_agree and prop_first ("simulation" or "expectation").
#' @param ntrials Numeric, number of trials per block (defaults to 1e4).
#'
#' @return Numeric, the MSE.
#'
#' @importFrom rlang .data
#'
#' @examples
#' \dontrun{
#' # generating prop_agree and prop_first from pars
#' sdt_data(pars = c(0, 1), ntrials = 1e4)
#' }
#'
#' @author Ladislas Nalborczyk \email{ladislas.nalborczyk@@gmail.com}.
#'
#' @references Goupil, L., Ponsot, E., Richardson, D. et al. (2021). Listeners'
#' perceptions of the certainty and honesty of a speaker are associated with a
#' common prosodic signature. *Nature Communications 12*, 861. \url{https://doi.org/10.1038/s41467-020-20649-4}.
#'
#' @export

sdt_data <- function (pars, method = c("simulation", "expectation"), ntrials = 1e4) {

    # some tests for variable types
    stopifnot("ntrials must be a numeric..." = is.numeric(ntrials) )

    # ensuring that the method is one of the above
    method <- match.arg(method)

    # following Goupil et al. (2021)'s notation,
    # s_i is the difference bw stim2 - stim1
    # sigma_ir is the difference bw noise of stim2 and noise of stim1
    # pc_agree is the predicted percentage of agreement
    # pc_int1 is the predicted probability (percentage) of chosen first stimuli

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
            # decision rule (from Goupil et al., 2021)
            dplyr::mutate(
                stim = dplyr::if_else(
                    condition = (.data$s_i + .data$sigma_ir) > bias,
                    true = "stim1", false = "stim2"
                    )
                ) |>
            # reshaping the dataframe
            tidyr::pivot_wider(
                names_from = .data$rep, values_from = .data$stim,
                id_cols = .data$trial
                )

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

            # handling the hedge case when noise = 0
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
#' @param log_mse Logcial, should we return the log-MSE (instead of the MSE).
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
#' @author Ladislas Nalborczyk \email{ladislas.nalborczyk@@gmail.com}.
#'
#' @export

sdt_loss <- function (pars, data, method = c("simulation", "expectation"), ntrials = 1e4, log_mse = TRUE) {

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
        # return (log_prediction_error)
        return (log(prediction_error) )

    } else {

        return (prediction_error)

    }

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
#' @param fit_method Character, the optimisation method, see possible values below (DEoptim seems to work best).
#' Beware that method "grid" can take some time, depending on the size of the grid.
#' @param cluster Character, existing parallel cluster object. If provided, overrides + specified parallelType.
#' @param grid_res Numeric, grid resolution in units of response bias or internal
#' noise (only used for method "grid").
#' @param maxit Numeric, maximum number of iterations.
#' @param verbose Logical, whether to print progress during fitting.
#' @param return_grid Logical, should we return the full grid when method = "grid".
#' @param smooth_grid Logical, should we smooth the error surface (grid) with a GAM.
#' @param smooth_k Numeric, k value in mgcv::gam() for smoothing the error surface.
#'
#' @return The optimised parameter values and further convergence information.
#'
#' @importFrom stats nlminb optim
#'
#' @examples
#' \dontrun{
#' # generating prop_agree and prop_first from pars
#' sdt_df <- sdt_data(pars = c(1, 2), ntrials = 1e4)
#'
#' # fitting the SDT model using bobyqa (fast method)
#' sdt_fitting(data = sdt_df, ntrials = 1e4, fit_method = "bobyqa")
#'
#' fitting the SDT model using DEoptim (slower but more accurate)
#' sdt_fit <- sdt_fitting(data = sdt_df, ntrials = 1e4, fit_method = "DEoptim")
#' summary(sdt_fit)
#'
#' fitting the SDT model using the grid method (super slower but accurate)
#' sdt_fitting(data = sdt_df, ntrials = 1e3, fit_method = "grid", grid_res = 0.1)
#' }
#'
#' @author Ladislas Nalborczyk \email{ladislas.nalborczyk@@gmail.com}.
#'
#' @export

sdt_fitting <- function (
        data,
        method = c("simulation", "expectation"),
        ntrials = 1e4,
        log_mse = TRUE,
        fit_method = c("DEoptim", "nlminb", "SANN", "Nelder-Mead", "CG", "BFGS", "bobyqa", "grid"),
        cluster = NULL,
        grid_res = 0.01,
        maxit = 100,
        verbose = FALSE,
        return_grid = FALSE,
        smooth_grid = TRUE,
        smooth_k = 10
        ) {

    # some tests for variable types
    stopifnot("data must be a dataframe..." = is.data.frame(data) )
    stopifnot("ntrials must be a numeric..." = is.numeric(ntrials) )
    stopifnot("grid_res must be a numeric..." = is.numeric(grid_res) )

    # method should be one of above
    method <- match.arg(method)

    # fit_method should be one of above
    fit_method <- match.arg(fit_method)

    if (fit_method == "DEoptim") {

        # starting the optimisation
        fit <- DEoptim::DEoptim(
            fn = sdt_loss,
            data = data,
            method = method,
            ntrials = ntrials,
            log_mse = log_mse,
            lower = c(-5, 0),
            upper = c(+5, 5),
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
                packages = c("DEoptim", "dplyr", "tidyr"),
                # defining the cluster
                cluster = cluster
                )
            )

    } else if (fit_method == "nlminb") {

        fit <- stats::nlminb(
            start = c(0, 1),
            objective = sdt_loss,
            data = data,
            ntrials = ntrials,
            log_mse = log_mse,
            lower = c(-5, 0),
            upper = c(+5, 5)
            )

    } else if (fit_method == "SANN") {

        fit <- stats::optim(
            par = c(bias = 0, noise = 1),
            fn = sdt_loss,
            data = data,
            ntrials = ntrials,
            log_mse = log_mse,
            method = fit_method
            )

    } else if (fit_method %in% c("Nelder-Mead", "CG", "BFGS", "bobyqa") ) {

        fit <- optimx::optimx(
            par = c(bias = 0, noise = 1),
            fn = sdt_loss,
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
            y = seq(from = 0, to = 5, by = grid_res)
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
                    palin::sdt_loss(x, data = data, ntrials = ntrials, log_mse = log_mse)
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

            # smoothing the error surface with a GAM
            message("Fitting a GAM to smooth the error function...")
            smoothing_model <- mgcv::gam(
                formula = z ~ te(x, y, k = smooth_k),
                data = param_grid
                )

            # making predictions about z
            param_grid$z_smoothed <- stats::fitted(smoothing_model)

            # finding the minimum (or minima) bias and noise values
            minima <- which(param_grid$z_smoothed == min(param_grid$z_smoothed) )

        }

        if (length(minima) == 1) {

            # if there is only one minimum, returns it
            avg_bias <- param_grid$x[minima]
            avg_noise <- param_grid$y[minima]
            fit <- data.frame(response_bias = avg_bias, internal_noise = avg_noise)

            } else {

                # otherwise, finds the average (median) bias and noise values across the minima
                message("Several minima found, returning the median parameter values...")
                avg_bias <- stats::median(param_grid$x[minima])
                avg_noise <- stats::median(param_grid$y[minima])
                fit <- data.frame(response_bias = avg_bias, internal_noise = avg_noise)

            }

        # should we return the full grid?
        if (return_grid) fit <- param_grid

    }

    # returning the fit
    return (fit)

}
