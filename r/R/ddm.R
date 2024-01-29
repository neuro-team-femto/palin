#' Log-likelihood of the DDM
#'
#' Computing the (negative) log-likelihood of the DDM using the `fddm` package.
#'
#' @param rt Numeric, the response time (in **seconds**).
#' @param resp Numeric, a vector indicating the chosen stimulus at each trial.
#' @param par Character, the optimisation method (i.e., "nlminb" or "optim").
#'
#' @return The negative log-likelihood.
#'
#' @examples
#' \dontrun{
#' # importing the data
#' data(self_produced_speech)
#' head(self_produced_speech)
#'
#' # reshaping the data
#' df <- self_produced_speech |>
#'   # removing the last block
#'   filter(block < 7) |>
#'   # keeping only the relevant columns
#'   select(participant, choice = response, RT) |>
#'   mutate(choice = ifelse(test = choice == "stim1", yes = 1, no = 2) )
#'
#' # splitting data by participant
#' df_ln <- df |> filter(participant == "LN")
#' df_mm <- df |> filter(participant == "MM")
#'
#' # computing the (negative) log-likelihood given some data and parameter values
#' # pars are a, v, t0, w, sv
#' ddm_log_likelihood(rt = df_ln$RT, resp = df_ln$choice, par = c(1, 1, 0, 0.5, 0) )
#' ddm_log_likelihood(rt = df_mm$RT, resp = df_mm$choice, par = c(1, 1, 0, 0.5, 0) )
#' }
#'
#' @author Ladislas Nalborczyk \email{ladislas.nalborczyk@@gmail.com}.
#'
#' @export

ddm_log_likelihood <- function (rt, resp, par) {

    dens <- fddm::dfddm(
        rt = rt, response = resp,
        a = par[[1]], v = par[[2]], t0 = par[[3]], w = par[[4]], sv = par[[5]],
        log = TRUE
        )

    return (ifelse(test = any(!is.finite(dens) ), yes = 1e6, no = -sum(dens) ) )

}

#' Fitting the full DDM
#'
#' Fitting the full DDM using the `fddm` package.
#'
#' @param rt Numeric, the response time (in **seconds**).
#' @param resp Numeric, a vector indicating the chosen stimulus at each trial.
#' @param method Character, the optimisation method (i.e., "nlminb" or "optim").
#'
#' @return The optimised parameter values and further output from nlminb() or optim().
#'
#' @importFrom stats nlminb optim
#' @export
#'
#' @examples
#' \dontrun{
#' # importing the data
#' data(self_produced_speech)
#' head(self_produced_speech)
#'
#' # reshaping the data
#' df <- self_produced_speech |>
#'   # removing the last block
#'   filter(block < 7) |>
#'   # keeping only the relevant columns
#'   select(participant, choice = response, RT) |>
#'   mutate(choice = ifelse(test = choice == "stim1", yes = 1, no = 2) )
#'
#' # splitting data by participant
#' df_ln <- df |> filter(participant == "LN")
#' df_mm <- df |> filter(participant == "MM")
#'
#' # fitting the full DDM for LN and MM (pars are a, v, t0, w, sv)
#' ddm_fitting(rt = df_ln$RT, resp = df_ln$choice, method = "nlminb")$par
#' ddm_fitting(rt = df_mm$RT, resp = df_mm$choice, method = "nlminb")$par
#' }

ddm_fitting <- function (rt, resp, method = c("nlminb", "optim") ) {

    if (method == "nlminb") {

        fit <- stats::nlminb(
            start = c(1, 1, 0, 0.5, 0),
            objective = ddm_log_likelihood,
            rt = rt, resp = resp,
            lower = c(0.01, -Inf, 0, 0, 0),
            upper = c(Inf, Inf, Inf, 1, Inf)
            )

        } else if (method == "optim") {

            fit <- stats::optim(
                par = c(1, 1, 0, 0.5, 0),
                fn = ddm_log_likelihood,
                rt = rt, resp = resp,
                method = "Nelder-Mead"
                )

            }

    return (fit)

}
