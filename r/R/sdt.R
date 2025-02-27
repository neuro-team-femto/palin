#' Generating data from the SDT model and computing the MSE
#'
#' Generating data from the SDT model and computing the MSE. Adapted from
#' Goupil et al. (2021), \url{https://doi.org/10.1038/s41467-020-20649-4}.
#'
#' @param pars Numeric, should be a list of initial values for the response bias
#' and the internal noise.
#' @param data Dataframe, with observed prop_agree and prop_int1.
#' @param ntrials Numeric, number of trials per block (defaults to 1e4).
#'
#' @return Numeric, the MSE.
#'
#' @importFrom rlang .data
#' @importFrom stats rnorm
#'
#' @examples
#' \dontrun{
#' # example data (70% agreement and no response bias)
#' df <- data.frame(prop_agree = 0.7, prop_int1 = 0.5, ntrials = 1e2)
#'
#' # computing the MSE
#' sdt(par = c(0, 1), data = df)
#' }
#'
#' @author Ladislas Nalborczyk \email{ladislas.nalborczyk@@gmail.com}.
#'
#' @references Goupil, L., Ponsot, E., Richardson, D. et al. (2021). Listeners'
#' perceptions of the certainty and honesty of a speaker are associated with a
#' common prosodic signature. *Nature Communications 12*, 861. \url{https://doi.org/10.1038/s41467-020-20649-4}.
#'
#' @export

sdt <- function (pars, data, ntrials = 1e4) {

    # some tests for variable types
    stopifnot("data must be a dataframe..." = is.data.frame(data) )
    stopifnot("ntrials must be a numeric..." = is.numeric(ntrials) )

    # following Goupil et al. (2021)'s notation,
    # s_i is the difference bw stim1 - stim1
    # sigma_ir is the difference bw noise of stim2 and noise of stim1
    # pc_agree is the predicted percentage of agreement
    # pc_int1 is the predicted probability (percentage) of chosen first stimuli

    # response bias
    bias <- pars[[1]]

    # internal noise
    noise <- pars[[2]]

    # simulating data
    simulated_df <- data.frame(
        # difference in representation between stimuli (stim2 - stim1)
        s_i = rep(x = stats::rnorm(n = ntrials, mean = 0, sd = 1), 2),
        # numbering trials for each block
        trial = 1:ntrials,
        # identifying the two (repeated) blocks
        rep = rep(c("block1", "block2"), each = ntrials)
        ) |>
        # difference in noise added to the stimuli (sigma_stim2 - sigma_stim1)
        dplyr::mutate(
            sigma_ir = stats::rnorm(n = ntrials * 2, mean = 0, sd = noise)
            ) |>
        # decision rule (from Goupil et al., 2021)
        dplyr::mutate(
            stim = dplyr::if_else(
                condition = (.data$s_i + .data$sigma_ir) < bias,
                true = "stim1", false = "stim2"
                )
            ) |>
        # reshaping the dataframe
        tidyr::pivot_wider(
            names_from = .data$rep, values_from = .data$stim,
            id_cols = .data$trial
            )

    # predicted percentage of agreement
    pc_agree <- sum(simulated_df$block1 == simulated_df$block2) / ntrials

    # predicted probability (percentage) of choosing the first stimulus
    pc_int1 <- (sum(simulated_df == "stim1") / ntrials) / 2

    # computing the squared prediction error (summed for the two percentages)
    prediction_error <- (data$prop_agree - pc_agree)^2 + (data$prop_int1 - pc_int1)^2

    # returning the MSE
    return (prediction_error)

}

#' Fitting the SDT model
#'
#' Fitting the SDT model. NB: Best results are obtained with the "DEoptim"
#' method and at least 1e4 trials.
#'
#' @param data Dataframe, with observed prop_agree and prop_int1.
#' @param ntrials Numeric, number of simulated trials in the SDT model.
#' @param method Character, the optimisation method, see possible values below (DEoptim seems to work best).
#' Beware that method "grid" can take some time, depending on the size of the grid.
#' @param cluster Character, existing parallel cluster object. If provided, overrides + specified parallelType.
#' @param grid_res Numeric, grid resolution in units of response bias or internal
#' noise (only used for method "grid").
#' @param maxit Numeric, maximum number of iterations.
#' @param verbose Boolean, whether to print progress during fitting.
#'
#' @return The optimised parameter values and further convergence information.
#'
#' @importFrom stats nlminb optim
#'
#' @examples
#' \dontrun{
#' # using summary statistics of empirical data
#' df <- data.frame(prop_agree = 0.7, prop_int1 = 0.413, ntrials = 1e2)
#'
#' # fitting the SDT model using bobyqa (fast method)
#' sdt_fitting(data = df, ntrials = 1e4, method = "bobyqa")
#'
#' fitting the SDT model using DEoptim (slower but more accurate)
#' sdt_fit <- sdt_fitting(data = df, ntrials = 1e4, method = "DEoptim")
#' summary(sdt_fit)
#'
#' fitting the SDT model using the grid method (super slower but more accurate)
#' sdt_fitting(data = df, ntrials = 1e3, method = "grid", grid_res = 0.05)
#' }
#'
#' @author Ladislas Nalborczyk \email{ladislas.nalborczyk@@gmail.com}.
#'
#' @export

sdt_fitting <- function (
        data,
        ntrials = 1e4,
        method = c("DEoptim", "nlminb", "SANN", "Nelder-Mead", "CG", "BFGS", "bobyqa", "grid"),
        cluster = NULL,
        grid_res = 0.01,
        maxit = 100,
        verbose = FALSE
        ) {

    # some tests for variable types
    stopifnot("data must be a dataframe..." = is.data.frame(data) )
    stopifnot("ntrials must be a numeric..." = is.numeric(ntrials) )
    stopifnot("grid_res must be a numeric..." = is.numeric(grid_res) )

    # method should be one of above
    method <- match.arg(method)

    if (method == "DEoptim") {

        # starting the optimisation
        fit <- DEoptim::DEoptim(
            fn = sdt,
            data = data,
            ntrials = ntrials,
            lower = c(-10, 0),
            upper = c(+10, 10),
            control = DEoptim::DEoptim.control(
                # maximum number of iterations
                itermax = maxit,
                # printing progress iteration
                trace = verbose,
                # printing progress every 10 iterations
                # trace = 10,
                # defines the differential evolution strategy (defaults to 2)
                # 1: DE / rand / 1 / bin (classical strategy)
                # 2: DE / local-to-best / 1 / bin (default)
                # 3: DE / best / 1 / bin with jitter
                # 4: DE / rand / 1 / bin with per-vector-dither
                # 5: DE / rand / 1 / bin with per-generation-dither
                # 6: DE / current-to-p-best / 1
                strategy = 3,
                # value to reach (defaults to -Inf)
                VTR = 0,
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
                # reltol = 1e-6,
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

    } else if (method == "nlminb") {

        fit <- stats::nlminb(
            start = c(0, 1),
            objective = sdt,
            data = data,
            ntrials = ntrials,
            lower = c(-Inf, 0),
            upper = c(Inf, Inf)
            )

    } else if (method == "SANN") {

        fit <- stats::optim(
            par = c(bias = 0, noise = 1),
            fn = sdt,
            data = data, ntrials = ntrials,
            method = method
            )

    } else if (method %in% c("Nelder-Mead", "CG", "BFGS", "bobyqa") ) {

        fit <- optimx::optimx(
            par = c(bias = 0, noise = 1),
            fn = sdt,
            data = data, ntrials = ntrials,
            method = method
            )

    } else if (method == "grid") {

        # starting parallel mode (by default using all available cores except one)
        future::plan(future::multisession(workers = parallel::detectCores() - 1) )

        # grid search: computing the MSE for many possible values of
        # response bias and internal noise (may take some time...)
        param_grid <- tidyr::crossing(
            x = seq(from = -5, to = 5, by = grid_res),
            y = seq(from = 0, to = 10, by = grid_res)
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
                    palin::sdt(x, data = data, ntrials = ntrials)
                    },
                future.seed = NULL
                )
            )

        # explicitly closing multisession workers by switching plan back to sequential
        future::plan(future::sequential)

        # finding the minimum (or minima) bias and noise values
        # we could also look for the minimum on a smoothed grid/surface
        minima <- which(param_grid$z == min(param_grid$z) )

        if (length(minima) == 1) {

            # if there is only one minimum, returns it
            avg_bias <- param_grid$x[minima]
            avg_noise <- param_grid$y[minima]
            fit <- data.frame(response_bias = avg_bias, internal_noise = avg_noise)

            } else {

                # otherwise, finds the average bias and noise values across the minima
                # NB: mean may be biased, better use the median?
                avg_bias <- mean(param_grid$x[minima])
                avg_noise <- mean(param_grid$y[minima])
                fit <- data.frame(response_bias = avg_bias, internal_noise = avg_noise)

            }

    }

    return (fit)

}
