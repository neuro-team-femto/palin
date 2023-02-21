#' Generating data from the SDT model and computing the MSE
#'
#' Generating data from the SDT model and computing the MSE. Adapted from
#' Goupil et al. (2021), \url{https://doi.org/10.1038/s41467-020-20649-4}.
#'
#' @param par Numeric, should be a list of initial values for the response bias
#' and the internal noise.
#' @param data Dataframe, with observed prop_agree and prop_int1.
#' @param ntrials Numeric, number of trials per block (defaults to 10k).
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
#' common prosodic signature. Nat Commun 12, 861. \url{https://doi.org/10.1038/s41467-020-20649-4}.
#'
#' @export

sdt <- function (par, data, ntrials = 1e4) {

    # some tests for variable types
    stopifnot("data must be a dataframe..." = is.data.frame(data) )
    stopifnot("ntrials must be a numeric..." = is.numeric(ntrials) )

    # following Goupil et al. (2021)'s notation,
    # s_i is the difference bw stim1 - stim1
    # sigma_ir is the difference bw noise of stim2 and noise of stim1
    # pc_agree is the predicted percentage of agreement
    # pc_int1 is the predicted probability (percentage) of chosen first stimuli

    # response bias
    bias <- par[[1]]

    # internal noise
    noise <- par[[2]]

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

#' Fitting SDT model
#'
#' Fitting the SDT model. NB: Best results are obtained with the "bobyqa"
#' method and at least 1e5 trials.
#'
#' @param data Dataframe, with observed prop_agree and prop_int1.
#' @param ntrials Numeric, number of trials in the SDT model.
#' @param method Character, the optimisation method, see possible values below.
#' Beware that method "grid" can take some time, depending on the size of the grid.
#' @param grid_res Numeric, grid resolution in units of response bias or internal
#' noise (only used for method "grid").
#'
#' @return The optimised parameter values and further convergence information.
#'
#' @details Add some doc here...
#'
#' @importFrom stats nlminb optim
#'
#' @examples
#' \dontrun{
#' # actual observations from pilot trials (LN and MM)
#' df_ln <- data.frame(prop_agree = 0.7, prop_int1 = 0.413, ntrials = 1e2)
#' df_mm <- data.frame(prop_agree = 0.58, prop_int1 = 0.563, ntrials = 1e2)
#'
#' # fitting the SDT model for LN and MM
#' sdt_fitting(data = df_ln, ntrials = 1e5, method = "bobyqa")
#' sdt_fitting(data = df_mm, ntrials = 1e5, method = "bobyqa")
#'
#' # for info: results obtained via the method of Ponsot (not in the package yet)
#' # for LN
#' # The minimised response bias value is: -0.227
#' # The minimised internal noise value is: 0.833
#' # for MM
#' # The minimised response bias value is: 0.227
#' # The minimised internal noise value is: 1.566
#' }
#'
#' @author Ladislas Nalborczyk \email{ladislas.nalborczyk@@gmail.com}.
#'
#' @export

sdt_fitting <- function (
        data, ntrials = 1e4,
        method = c("nlminb", "SANN", "Nelder-Mead", "CG", "BFGS", "bobyqa", "grid"),
        grid_res = 0.01
        ) {

    # some tests for variable types
    stopifnot("data must be a dataframe..." = is.data.frame(data) )
    stopifnot("ntrials must be a numeric..." = is.numeric(ntrials) )
    stopifnot("grid_res must be a numeric..." = is.numeric(grid_res) )

    # method should be one of above
    method <- match.arg(method)

    if (method == "nlminb") {

        fit <- stats::nlminb(
            start = c(0, 1),
            objective = sdt,
            data = data, ntrials = ntrials,
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

        # opening a parallel cluster (by default using all available clusters)
        cl <- parallel::makeCluster(parallel::detectCores() / 2)
        parallel::clusterEvalQ(cl = cl, expr = {library(tidyverse)})

        # grid search: computing the MSE for many possible values of
        # response bias and internal noise (may take some time...)
        param_grid <- tidyr::crossing(
            x = seq(from = -5, to = 5, by = grid_res),
            y = seq(from = 0, to = 10, by = grid_res)
            )

        param_grid <- dplyr::mutate(
            .data = param_grid,
            z = parallel::parApply(
                cl = cl, X = param_grid,
                MARGIN = 1, FUN = palin::sdt, data = data, ntrials = ntrials
                )
            )

        # how long did it take?
        # (using 8 cores, ~40 secs for a 100x100 grid, ~55 min for a 1000x1000 grid)
        # (~90 min for a 1000x1000 grid with 10e4 trials)
        # end_time <- Sys.time()
        # print(end_time - start_time)

        # stopping (closing) the cluster
        parallel::stopCluster(cl)

        # finding the minimum (or minima) bias and noise values
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
                # fit <- data.frame(t(colMeans(param_grid[minima, ]) ) )
                fit <- data.frame(response_bias = avg_bias, internal_noise = avg_noise)

            }

    }

    return (fit)

}
