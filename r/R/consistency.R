#' Computing different metrics of response's consistency.
#'
#' Computing different metrics of response's consistency from both double-pass
#' and non double-pass reverse correlation data.
#'
#' @param data Dataframe, with reverse correlation data (in long format).
#' @param participant_id Numeric/Character/Factor, column in data specifying the participant ID.
#' @param block_id Numeric, column in data specifying the block ID.
#' @param trial_id Numeric, column in data specifying the trial ID.
#' @param feature_id Numeric/Factor, column in data specifying the feature.
#' @param value_id Numeric, column in data specifying the feature value
#' @param response_id Numeric, column in data specifying the response.
#' @param method Character, which method to use (distance to templates or similarity to kernel).
#' @param double_pass Logical, indicating whether the last block was repeated.
#'
#' @return Dataframe, various metrices of response consistency.
#'
#' @importFrom rlang .data
#' @importFrom utils tail
#'
#' @examples
#' \dontrun{
#' # importing the self-voice data
#' data(self_voice)
#' head(self_voice)
#'
#' # computing metrics of response consistency per participant and per block
#' response_consistency(self_voice, method = "template_distance") |> head(10)
#' response_consistency(self_voice, method = "kernel_similarity") |> head(10)
#' }
#'
#' @author Ladislas Nalborczyk \email{ladislas.nalborczyk@@gmail.com}.
#'
#' @export

response_consistency <- function (
        data,
        participant_id = "participant", block_id = "block", trial_id = "trial",
        feature_id = "feature", value_id = "value", response_id = "response",
        method = c("template_distance", "kernel_similarity"),
        double_pass = TRUE
        ) {

    # some tests for variable types
    stopifnot("data must be a dataframe..." = is.data.frame(data) )

    # method should be one of above
    method <- match.arg(method)

    # checking required column names
    required_columns <- c(participant_id, block_id, trial_id, feature_id, value_id, response_id)
    assertthat::assert_that(
        all(required_columns %in% colnames(data) ),
        msg = paste(
            "Missing columns:",
            paste(setdiff(required_columns, colnames(data) ), collapse = ", ")
            )
        )

    # computing first-order kernel for each participant
    kernels <- computing_kernel(
        data,
        participant_id = participant_id, block_id = block_id,
        trial_id = trial_id, feature_id = feature_id,
        value_id = value_id, response_id = response_id,
        method = "difference"
        )

    if (method == "template_distance") {

        # computing average (and distance-weighted) consistency
        consistency <- data |>
            dplyr::left_join(
                dplyr::select(
                    kernels, .data[[participant_id]], .data[[feature_id]],
                    .data$negative, .data$positive
                    ),
                by = c(participant_id, feature_id)
                ) |>
            # removing unused columns
            dplyr::select(-.data[[feature_id]]) |>
            # removing the last block (double pass)
            dplyr::filter(if (double_pass) .data[[block_id]] < max(.data[[block_id]]) else TRUE) |>
            # grouping per participant, block, trial, and stimulus
            dplyr::group_by(
                .data[[participant_id]], .data[[block_id]],
                .data[[trial_id]], .data[[response_id]]
                ) |>
            # computing the Euclidean distance from the average chosen (positive)
            # and not-chosen (negative) position in parameters' space
            dplyr::summarise(
                distance_from_positive = sqrt(sum((.data[[value_id]] - .data$positive)^2) ),
                distance_from_negative = sqrt(sum((.data[[value_id]] - .data$negative)^2) )
                ) |>
            dplyr::ungroup() |>
            # removing the stimuli that were not chosen
            dplyr::filter(.data[[response_id]] == 1) |>
            # grouping per participant, block, trial, and response
            dplyr::group_by(
                .data[[participant_id]], .data[[block_id]],
                .data[[trial_id]], .data[[response_id]]
                ) |>
            # computing consistency
            dplyr::mutate(
                consistency = as.numeric(
                    .data$distance_from_positive < .data$distance_from_negative
                    )
                ) |>
            dplyr::ungroup() |>
            # removing any NAs
            stats::na.omit() |>
            # grouping per participant and block
            dplyr::group_by(.data[[participant_id]], .data[[block_id]]) |>
            # computing average and weighted consistency
            dplyr::summarise(
                avg_consistency = mean(.data$consistency),
                weighted_consistency = stats::weighted.mean(
                    x = .data$consistency,
                    w = (.data$distance_from_negative / .data$distance_from_positive) /
                        sum(.data$distance_from_negative / .data$distance_from_positive)
                    )
                ) |>
            dplyr::ungroup() |>
            data.frame()

    } else if (method == "kernel_similarity") {

        # computing kernel similarity and consistency
        consistency <- data |>
            dplyr::left_join(
                dplyr::select(
                    kernels, .data[[participant_id]], .data[[feature_id]],
                    .data$negative, .data$positive
                    ),
                by = c(participant_id, feature_id)
                ) |>
            # removing unused columns
            dplyr::select(-.data[[feature_id]]) |>
            # removing the last block (double pass)
            dplyr::filter(if (double_pass) .data[[block_id]] < max(.data[[block_id]]) else TRUE) |>
            # grouping per participant, block, trial, and stimulus
            dplyr::group_by(
                .data[[participant_id]], .data[[block_id]],
                .data[[trial_id]], .data[[response_id]]
                ) |>
            # computing the similarity to kernel for each stimulus
            # dot product is a %*% b, or just sum(a*b)
            dplyr::mutate(kernel = .data$positive - .data$negative) |>
            dplyr::summarise(
                kernel_similarity = sum(.data[[value_id]] * .data$kernel),
                ) |>
            dplyr::ungroup() |>
            tidyr::pivot_wider(names_from = .data[[response_id]], values_from = .data$kernel_similarity) |>
            # renaming the columns
            dplyr::rename(negative = .data$`0`, positive = .data$`1`) |>
            # grouping per participant, block, trial, and response
            dplyr::group_by(
                .data[[participant_id]], .data[[block_id]],
                .data[[trial_id]],
                ) |>
            # computing consistency
            dplyr::mutate(
                consistency = as.numeric(
                    .data$positive > .data$negative
                    )
                ) |>
            dplyr::ungroup() |>
            # removing any NAs
            stats::na.omit() |>
            dplyr::mutate(positive_negative_diff = .data$positive - .data$negative) |>
            # grouping per participant and block
            dplyr::group_by(.data[[participant_id]], .data[[block_id]]) |>
            # computing average and weighted consistency
            dplyr::summarise(
                avg_consistency = mean(.data$consistency),
                weighted_consistency = stats::weighted.mean(
                    x = .data$consistency,
                    w = (.data$positive_negative_diff - min(.data$positive_negative_diff) ) /
                        (max(.data$positive_negative_diff) - min(.data$positive_negative_diff) )
                    )
                ) |>
            dplyr::ungroup() |>
            data.frame()

    }

    if (double_pass == TRUE) {

        # retrieving the number of blocks
        unique_blocks <- unique(data[[block_id]])

        # identifying double-pass block (by default the last two blocks)
        dp_blocks <- utils::tail(x = unique_blocks, n = 2)

        # computing the percentage of first stim chosen in the two double-pass blocks
        double_pass_prop_first <- data |>
            dplyr::filter(if (double_pass) .data[[block_id]] %in% dp_blocks else TRUE) |>
            # adding a variable to indicate stimulus position
            dplyr::mutate(
                stim = cumsum(c(1, diff(.data$response) != 0) ),
                .by = c(.data[[participant_id]], .data[[block_id]], .data[[trial_id]])
                ) |>
            dplyr::select(
                .data[[participant_id]], .data[[trial_id]],
                .data[[block_id]], .data[[response_id]], .data$stim
                ) |>
            dplyr::group_by(.data[[participant_id]]) |>
            dplyr::filter(.data[[response_id]] == 1) |>
            dplyr::mutate(
                pfirst = dplyr::if_else(.data$stim == 1 & .data[[response_id]] == 1, 1, 0)
                ) |>
            dplyr::summarise(pfirst = mean(.data$pfirst) ) |>
            dplyr::ungroup() |>
            data.frame()

        # computing the percentage of agreement in the two double-pass blocks
        double_pass_prop_agree <- data |>
            dplyr::filter(if (double_pass) .data[[block_id]] %in% dp_blocks else TRUE) |>
            dplyr::select(-.data[[feature_id]], -.data[[value_id]]) |>
            dplyr::distinct() |>
            # reshaping the response variable as indicating int1 or int2
            dplyr::mutate(
                resp = dplyr::if_else(dplyr::first(.data[[response_id]]) == 1, 0, 1),
                .by = c(.data[[participant_id]], .data[[trial_id]])
                ) |>
            # removing unused columns
            dplyr::select(.data[[participant_id]], .data[[block_id]], .data[[trial_id]], .data$resp) |>
            # removing duplicated rows
            dplyr::distinct() |>
            # reshaping the trial variable
            dplyr::mutate(trial = 1:dplyr::n(), .by = c(.data[[participant_id]], .data[[block_id]]) ) |>
            tidyr::pivot_wider(
                names_from = .data[[block_id]],
                # values_from = .data[[response_id]],
                values_from = .data$resp,
                names_prefix = "block"
                ) |>
            # computing agreement
            dplyr::mutate(
                agreement = dplyr::if_else(
                    dplyr::cur_data()[[ncol(dplyr::cur_data()) - 1]] ==
                        dplyr::cur_data()[[ncol(dplyr::cur_data())]],
                    1, 0
                    )
                ) |>
            # dplyr::mutate(
            #     agreement = dplyr::if_else(
            #         dplyr::cur_data()[[dp_blocks[1]]] ==
            #             dplyr::cur_data()[[dp_blocks[2]]],
            #         1, 0
            #         )
            #     ) |>
            # remove potential NA
            stats::na.omit() |>
            dplyr::summarise(
                double_pass_prop_agree = mean(.data$agreement),
                .by = .data[[participant_id]]
                ) |>
            data.frame()

        # including this metric in the consistency dataframe
        consistency <- dplyr::left_join(
            consistency,
            double_pass_prop_agree,
            # by = .data[[participant_id]]
            by = participant_id
            )

        consistency <- dplyr::left_join(
            consistency,
            double_pass_prop_first,
            # by = .data[[participant_id]]
            by = participant_id
            )

    }

    # returning the consistency metrics
    return (consistency)

}
