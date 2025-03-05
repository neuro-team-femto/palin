#' Computing different metrics of response's consistency.
#'
#' Computing different metrics of response's consistency from both double-pass
#' and non double-pass reverse correlation data.
#'
#' @param data Dataframe, with reverse correlation data. Should contain the
#' following columns: participant, block, trial, resp, f, eq.
#' @param double_pass Logical, indicating whether the last block was repeated.
#'
#' @return Numeric, the MSE.
#'
#' @importFrom rlang .data
#'
#' @examples
#' \dontrun{
#' # importing the self-voice data
#' data(self_voice)
#' head(self_voice)
#'
#' # computing metrics of response consistency per participant and per block
#' response_consistency(self_voice) |> head(10)
#' }
#'
#' @author Ladislas Nalborczyk \email{ladislas.nalborczyk@@gmail.com}.
#'
#' @export

response_consistency <- function (data, double_pass = TRUE) {

    # some tests for variable types
    stopifnot("data must be a dataframe..." = is.data.frame(data) )

    # computing first-order kernel for each participant
    kernels <- computing_kernel(data)

    # computing average (and distance-weighted) consistency
    consistency <- data |>
        dplyr::left_join(
            dplyr::select(kernels, .data$participant, .data$f, .data$negative, .data$positive),
            by = c("participant", "f")
            ) |>
        # removing unused columns
        dplyr::select(-.data$RT, -.data$f) |>
        # removing the last block (double pass)
        dplyr::filter(if (double_pass) .data$block < max(.data$block) else TRUE) |>
        # grouping per participant, block, trial, and stimulus
        dplyr::group_by(.data$participant, .data$block, .data$trial, .data$resp) |>
        # computing the Euclidean distance from the average chosen (positive)
        # and not-chosen (negative) position in parameters' space
        dplyr::summarise(
            distance_from_positive = sqrt(sum((.data$eq - .data$positive)^2) ),
            distance_from_negative = sqrt(sum((.data$eq - .data$negative)^2) )
            ) |>
        dplyr::ungroup() |>
        # removing the stimuli that were not chosen
        dplyr::filter(.data$resp == 1) |>
        # grouping per participant, block, trial, and stimulus
        dplyr::group_by(.data$participant, .data$block, .data$trial, .data$resp) |>
        # computing consistency
        dplyr::mutate(
            consistency = as.numeric(
                .data$distance_from_positive < .data$distance_from_negative
                )
            ) |>
        dplyr::ungroup() |>
        # grouping per participant and block
        dplyr::group_by(.data$participant, .data$block) |>
        # computing average and weighted consistency
        dplyr::summarise(
            avg_consistency = mean(.data$consistency),
            weighted_avg_consistency = stats::weighted.mean(
                x = .data$consistency,
                w = (.data$distance_from_negative / .data$distance_from_positive) /
                    sum(.data$distance_from_negative / .data$distance_from_positive)
                )
            ) |>
        dplyr::ungroup() |>
        data.frame()

    if (double_pass == TRUE) {

        # computing the percentage of agreement in the two double-pass blocks
        double_pass_prop_agree <- data |>
            # dplyr::filter(.data$block %in% c(max(.data$block)-1, max(.data$block) ) ) |>
            dplyr::filter(if (double_pass) .data$block < max(.data$block) else TRUE) |>
            dplyr::select(-.data$RT, -.data$f, -.data$eq) |>
            dplyr::distinct() |>
            # reshaping the response variable as indicating int1 or int2
            dplyr::mutate(
                resp = dplyr::if_else(dplyr::first(.data$resp) == 1, 0, 1),
                .by = c(.data$participant, .data$trial)
                ) |>
            # removing duplicated rows
            dplyr::distinct() |>
            # reshaping the trial variable
            dplyr::mutate(trial = 1:dplyr::n(), .by = c(.data$participant, .data$block) ) |>
            tidyr::pivot_wider(names_from = .data$block, values_from = .data$resp, names_prefix = "block") |>
            # computing agreement
            dplyr::mutate(agreement = dplyr::if_else(dplyr::pick(3) == dplyr::pick(4), 1, 0) ) |>
            dplyr::summarise(double_pass_prop_agree = mean(.data$agreement), .by = .data$participant) |>
            data.frame()

        # including this metric in the consistency dataframe
        consistency <- dplyr::left_join(consistency, double_pass_prop_agree, by = "participant")

    }

    # returning the consistency metrics
    return (consistency)

}
