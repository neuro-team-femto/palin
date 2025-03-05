#' Log-likelihood of the DDM
#'
#' Computing the (negative) log-likelihood of the DDM using the `fddm` package.
#'
#' @param pars Numeric, should be a list of initial values for the parameters.
#' @param rt Numeric, the response time (in **seconds**).
#' @param resp Numeric, a vector indicating the chosen stimulus at each trial.
#'
#' @return The negative log-likelihood.
#'
#' @examples
#' \dontrun{
#' # importing the self-voice data
#' data(self_voice)
#' head(self_voice)
#'
#' # reshaping the data
#' df <- self_voice |>
#'   # removing the last block
#'   filter(block < max(block) ) |>
#'   # keeping only the relevant columns
#'   select(participant, choice = resp, RT) |>
#'   mutate(choice = ifelse(test = choice == "stim1", yes = 1, no = 2) )
#'
#' # splitting data by participant (keeping only the first one)
#' df_ppt <- df |> filter(participant == unique(participant)[1])
#'
#' # computing the (negative) log-likelihood given some data and parameter values
#' # pars are a, v, t0, w, sv
#' ddm_log_likelihood(rt = df_ppt$RT, resp = df_ppt$choice, par = c(1, 1, 0, 0.5, 0) )
#' }
#'
#' @author Ladislas Nalborczyk \email{ladislas.nalborczyk@@gmail.com}.
#'
#' @export

ddm_log_likelihood <- function (pars, rt, resp) {

    # computing the DDM log-likelihood
    # see https://cran.rstudio.org/web/packages/fddm/vignettes/example.html
    dens <- fddm::dfddm(
        rt = rt, response = resp,
        v = pars[[1]], a = pars[[2]], t0 = pars[[3]], w = pars[[4]], sv = pars[[5]],
        err_tol = 1e-6, log = TRUE
        )

    # returning the negative log-likelihood
    return (ifelse(test = any(!is.finite(dens) ), yes = 1e6, no = -sum(dens) ) )

}

#' Fitting the full DDM
#'
#' Fitting the full DDM using the `fddm` package.
#'
#' @param rt Numeric, the response time (in **seconds**).
#' @param resp Numeric, a vector indicating the chosen stimulus at each trial.
#' @param method Character, the optimisation method (i.e., "nlminb", "optim", or "DEoptim").
#' @param maxit Numeric, maximum number of iterations.
#' @param cluster Character, existing parallel cluster object. If provided, overrides + specified parallelType.
#' @param verbose Boolean, whether to print progress during fitting.
#'
#' @return The optimised parameter values and further optimisation output.
#'
#' @importFrom stats nlminb optim
#' @export
#'
#' @examples
#' \dontrun{
#' # importing the data
#' data(self_voice)
#' head(self_voice)
#'
#' # reshaping the data
#' df <- self_voice |>
#'   # removing the last block
#'   filter(block < 7) |>
#'   # keeping only the relevant columns
#'   select(participant, choice = resp, RT) |>
#'   mutate(choice = ifelse(test = choice == "stim1", yes = 1, no = 2) )
#'
#' # splitting data by participant (keeping only the first one)
#' df_ppt <- df |> filter(participant == unique(participant)[1])
#'
#' # fitting the full DDM for LN and MM (pars are a, v, t0, w, sv)
#' ddm_fitting(rt = df_ppt$RT, resp = df_ppt$choice, method = "nlminb")$par
#' }

ddm_fitting <- function (
        rt, resp,
        method = c("nlminb", "optim", "DEoptim"),
        maxit = 1e3,
        cluster = NULL,
        verbose = FALSE
        ) {

    if (method == "nlminb") {

        fit <- stats::nlminb(
            start = c(0, 0.1, 0.1, 0.5, 0.1),
            objective = ddm_log_likelihood,
            rt = rt, resp = resp,
            lower = c(-5, 0, 0, 0, 0),
            upper = c(+5, 5, 5, 1, 5)
            )

        } else if (method == "optim") {

            fit <- stats::optim(
                par = c(0, 0.1, 0.1, 0.5, 0.1),
                fn = ddm_log_likelihood,
                rt = rt, resp = resp,
                method = "Nelder-Mead"
                )

            } else if (method == "DEoptim") {

                # starting the optimisation
                fit <- DEoptim::DEoptim(
                    fn = ddm_log_likelihood,
                    rt = rt, resp = resp,
                    lower = c(-5, 0, 0, 0, 0),
                    upper = c(+5, 5, 5, 1, 5),
                    control = DEoptim::DEoptim.control(
                        # maximum number of iterations
                        itermax = maxit,
                        # printing progress
                        trace = verbose,
                        # value to reach (defaults to -Inf)
                        # VTR = 0,
                        # using all available cores by default
                        parallelType = "parallel",
                        # defining the package to be imported on each parallel core
                        packages = c("DEoptim", "dplyr", "tidyr", "fddm"),
                        # defining the cluster
                        cluster = cluster
                        )
                    )

                }

    return (fit)

}
