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
#' @param response_id Numeric, column in data specifying the response (should be in the 0/1 format).
#' @param method Character, which consistency method to use. Options are
#'   `"template_distance"`, `"kernel_similarity"`, `"intercept"`,
#'   `"calibrated_kernel"`, `"calibrated_template"`, and `"full_glm"`.
#'   The calibrated methods estimate trial-wise response probabilities and
#'   convert them into expected double-pass agreement.
#' @param double_pass Logical, indicating whether the last block was repeated.
#' @param link_function Character, which binomial GLM link function to use in the "intercept" method (see ?binomial).
#' @param return_by Character. Should metrics be returned per participant and
#'   block (`"block"`) or pooled across all eligible blocks per participant
#'   (`"all"`)? Defaults to `"all"`.
#'
#' @return Dataframe, various metrics of response consistency.
#'
#' @importFrom rlang .data
#'
#' @examples
#' \dontrun{
#' # import the self-voice data
#' data(self_voice)
#' head(self_voice)
#'
#' # compute response consistency per participant
#' response_consistency(self_voice, method = "template_distance") |> head(10)
#' response_consistency(self_voice, method = "kernel_similarity") |> head(10)
#' response_consistency(self_voice, method = "intercept") |> head(10)
#' response_consistency(self_voice, method = "intercept", link_function = "probit") |> head(10)
#' response_consistency(self_voice, method = "calibrated_kernel")
#' response_consistency(self_voice, method = "calibrated_template")
#' response_consistency(self_voice, method = "full_glm")
#'
#' # compute response consistency per participant and block
#' response_consistency(self_voice, method = "full_glm", return_by = "block")
#' }
#'
#' @author Ladislas Nalborczyk \email{ladislas.nalborczyk@@cnrs.fr}.
#'
#' @export

response_consistency <- function (
        data,
        participant_id = "participant", block_id = "block", trial_id = "trial",
        feature_id = "feature", value_id = "value", response_id = "response",
        method = c(
            "calibrated_template",
            "calibrated_kernel",
            "full_glm",
            "intercept",
            "template_distance",
            "kernel_similarity"
            ),
        double_pass = TRUE,
        link_function = c("logit", "probit"),
        return_by = c("all", "block")
        ) {

    make_trial_level_data <- function (data) {

        trial_df <- data |>
            dplyr::filter(
                if (double_pass) .data[[block_id]] < max(.data[[block_id]]) else TRUE
                ) |>
            dplyr::mutate(
                stim = cumsum(c(1, diff(.data[[response_id]]) != 0)),
                .by = c(
                    .data[[participant_id]],
                    .data[[block_id]],
                    .data[[trial_id]]
                    )
                ) |>
            dplyr::group_by(
                .data[[participant_id]],
                .data[[block_id]],
                .data[[trial_id]],
                .data[[feature_id]]
                ) |>
            dplyr::reframe(
                value_vector =
                    .data[[value_id]][.data$stim == 2] -
                    .data[[value_id]][.data$stim == 1],
                choice_stim2 = as.integer(
                    .data[[response_id]][.data$stim == 2][1] == 1
                    )
                ) |>
            dplyr::ungroup() |>
            tidyr::pivot_wider(
                names_from = .data[[feature_id]],
                values_from = .data$value_vector,
                names_prefix = "feature_"
                )

        return (trial_df)

    }

    # some tests for variable types
    stopifnot("data must be a dataframe..." = is.data.frame(data) )

    # ensuring that the method is one of the above
    method <- match.arg(method)
    link_function <- match.arg(link_function)
    return_by <- match.arg(return_by)

    # define summary grouping
    summary_groups <- if (return_by == "block") {

        c(participant_id, block_id)

    } else {

        participant_id

    }

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
        method = "difference",
        double_pass = double_pass
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
            # dplyr::filter(.data[[response_id]] == 1) |>
            # grouping per participant, block, trial, and response
            # dplyr::group_by(
            #     .data[[participant_id]], .data[[block_id]],
            #     .data[[trial_id]], .data[[response_id]]
            #     ) |>
            tidyr::pivot_wider(
                names_from = .data[[response_id]],
                values_from = c(.data$distance_from_positive, .data$distance_from_negative)
                ) |>
            dplyr::group_by(
                .data[[participant_id]], .data[[block_id]], .data[[trial_id]],
                ) |>
            # computing consistency
            # dplyr::mutate(
            #     consistency = as.numeric(
            #         .data$distance_from_positive < .data$distance_from_negative
            #         )
            #     ) |>
            # computing consistency as the ratio of ratios of distance for each stimulus
            dplyr::mutate(
                consistency = as.numeric(
                    ((.data$distance_from_negative_1 / .data$distance_from_positive_1) / (.data$distance_from_negative_0 / .data$distance_from_positive_0) ) > 1
                    # or defining consistency with 2 conditions
                    # .data$distance_from_positive_1 < .data$distance_from_negative_1 & .data$distance_from_positive_1 < .data$distance_from_positive_0
                    # equivalent to
                    # (.data$distance_from_negative_1 / .data$distance_from_positive_1) > 1 & (.data$distance_from_positive_0 / .data$distance_from_positive_1) > 1
                    )
                ) |>
            dplyr::ungroup() |>
            # removing any NAs
            stats::na.omit() |>
            # grouping per participant and block
            # dplyr::group_by(.data[[participant_id]], .data[[block_id]]) |>
            # computing average and weighted consistency
            dplyr::summarise(
                avg_consistency = mean(.data$consistency),
                # weighted_consistency = stats::weighted.mean(
                #     x = .data$consistency,
                #     w = (.data$distance_from_negative / .data$distance_from_positive) /
                #         sum(.data$distance_from_negative / .data$distance_from_positive)
                #     )
                .by = dplyr::all_of(summary_groups)
                ) |>
            # dplyr::ungroup() |>
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
            # reshaping the response column in 0/1 (if needed)
            dplyr::mutate(response_internal = as.numeric(as.factor(.data[[response_id]]) )-1) |>
            dplyr::select(-.data[[response_id]]) |>
            # tidyr::pivot_wider(names_from = .data[[response_id]], values_from = .data$kernel_similarity) |>
            tidyr::pivot_wider(names_from = .data$response_internal, values_from = .data$kernel_similarity) |>
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
            # dplyr::group_by(.data[[participant_id]], .data[[block_id]]) |>
            # computing average and weighted consistency
            dplyr::summarise(
                avg_consistency = mean(.data$consistency),
                # weighted_consistency = stats::weighted.mean(
                #     x = .data$consistency
                #     w = (.data$positive_negative_diff - min(.data$positive_negative_diff) ) /
                #         (max(.data$positive_negative_diff) - min(.data$positive_negative_diff) )
                #     )
                .by = dplyr::all_of(summary_groups)
                ) |>
            # dplyr::ungroup() |>
            data.frame()

    } else if (method == "intercept") {

        # retrieving the responses per participant and trial (first or second stimulus?)
        response_lookup <- data |>
            dplyr::filter(if (double_pass) .data[[block_id]] < max(.data[[block_id]]) else TRUE) |>
            # dplyr::select(.data[[participant_id]], .data[[trial_id]], .data[[response_id]]) |>
            dplyr::select(.data[[participant_id]], .data[[block_id]], .data[[trial_id]], .data[[response_id]]) |>
            dplyr::distinct() |>
            # reshaping the response variable as indicating int1 or int2
            dplyr::mutate(
                response = dplyr::if_else(dplyr::first(.data[[response_id]]) == 1, 0, 1),
                # .by = c(.data[[participant_id]], .data[[trial_id]])
                .by = c(.data[[participant_id]], .data[[block_id]], .data[[trial_id]])
                ) |>
            dplyr::distinct()

        # pivoting data to one entry per (trial, stim)
        value_lookup <- data |>
            # removing the last block (double pass)
            dplyr::filter(if (double_pass) .data[[block_id]] < max(.data[[block_id]]) else TRUE) |>
            # adding a variable to indicate stimulus position
            dplyr::mutate(
                # stim = cumsum(c(1, diff(.data$response) != 0) ),
                stim = cumsum(c(1, diff(.data[[response_id]]) != 0) ),
                .by = c(.data[[participant_id]], .data[[block_id]], .data[[trial_id]])
                ) |>
            # computing the feature difference for this trial
            dplyr::group_by(.data[[participant_id]], .data[[trial_id]], .data[[feature_id]]) |>
            # dplyr::summarise(
            #     value_vector = .data[[value_id]][.data$stim==2]-.data[[value_id]][.data$stim==1],
            #     .groups = "drop"
            #     ) |>
            dplyr::reframe(
                value_vector = .data[[value_id]][.data$stim==2] -
                    .data[[value_id]][.data$stim==1]
                ) |>
            # retrieving the kernels
            dplyr::left_join(
                kernels |> dplyr::select(
                    .data[[participant_id]], .data[[feature_id]],
                    .data$negative, .data$positive
                    ),
                by = c(participant_id, feature_id)
                ) |>
            dplyr::mutate(
                value_vector_kernel = sum(.data$value_vector * (.data$positive - .data$negative) ),
                .by = c(.data[[participant_id]], .data[[trial_id]])
                ) |>
            # removing some columns
            dplyr::select(-.data[[feature_id]], -.data$value_vector, -.data$negative, -.data$positive) |>
            # retrieving the response
            dplyr::left_join(response_lookup, by = c(participant_id, trial_id) ) |>
            # removing duplicated rows
            dplyr::distinct()

        # generating all trial pairs
        # trial_pairs <- tidyr::crossing(
        #     participant = unique(value_lookup[[participant_id]]),
        #     trial_1 = unique(value_lookup[[trial_id]]),
        #     trial_2 = unique(value_lookup[[trial_id]])
        #     ) |>
        #     # keeping unique pairs only
        #     dplyr::filter(.data$trial_1 < .data$trial_2)

        # computing difference between trial pairs
        # maybe check https://multidplyr.tidyverse.org/articles/multidplyr.html
        message ("Computing trial similarity for all pairs of trials, this may take some time...\n")

        # comb_df <- trial_pairs |>
        #     dplyr::left_join(
        #         value_lookup,
        #         by = c("participant", "trial_1" = trial_id)
        #         ) |>
        #     # dplyr::rename(value_1 = .data$value_vector_kernel, response_1 = .data$response) |>
        #     dplyr::rename(value_1 = .data$value_vector_kernel, response_1 = .data[[response_id]]) |>
        #     dplyr::left_join(
        #         value_lookup,
        #         by = c("participant", "trial_2" = trial_id)
        #         ) |>
        #     # dplyr::rename(value_2 = .data$value_vector_kernel, response_2 = .data$response) |>
        #     dplyr::rename(value_2 = .data$value_vector_kernel, response_2 = .data[[response_id]]) |>
        #     dplyr::rowwise() |>
        #     dplyr::mutate(
        #         value_diff = .data$value_1 - .data$value_2,
        #         rms_distance = sqrt(sum(.data$value_diff)^2),
        #         agree = as.integer(.data$response_1 == .data$response_2)
        #         ) |>
        #     dplyr::ungroup()

        value_lookup_split <- value_lookup |>
            dplyr::group_by(dplyr::across(dplyr::all_of(summary_groups) ) ) |>
            tidyr::nest()

        # comb_df <- value_lookup_split |>
        #     dplyr::mutate(
        #         pair_data = purrr::map(
        #             .data$data,
        #             function (df) {
        #
        #                 trial_pairs <- tidyr::crossing(
        #                     trial_1 = unique(df[[trial_id]]),
        #                     trial_2 = unique(df[[trial_id]])
        #                 ) |>
        #                     dplyr::filter(.data$trial_1 < .data$trial_2)
        #
        #                 trial_pairs |>
        #                     dplyr::left_join(
        #                         df,
        #                         by = c("trial_1" = trial_id)
        #                     ) |>
        #                     dplyr::rename(
        #                         value_1 = .data$value_vector_kernel,
        #                         response_1 = .data$response
        #                     ) |>
        #                     dplyr::left_join(
        #                         df,
        #                         by = c("trial_2" = trial_id)
        #                     ) |>
        #                     dplyr::rename(
        #                         value_2 = .data$value_vector_kernel,
        #                         response_2 = .data$response
        #                     ) |>
        #                     dplyr::mutate(
        #                         rms_distance = abs(.data$value_1 - .data$value_2),
        #                         agree = as.integer(.data$response_1 == .data$response_2)
        #                     )
        #
        #             }
        #         )
        #     ) |>
        #     dplyr::select(-.data$data) |>
        #     tidyr::unnest(.data$pair_data)

        comb_df <- value_lookup_split |>
            dplyr::mutate(
                pair_data = purrr::map(
                    .data$data,
                    function (df) {

                        trial_pairs <- tidyr::crossing(
                            trial_1 = unique(df[[trial_id]]),
                            trial_2 = unique(df[[trial_id]])
                        ) |>
                            dplyr::filter(.data$trial_1 < .data$trial_2)

                        trial_pairs |>
                            dplyr::left_join(
                                df,
                                by = c("trial_1" = trial_id)
                            ) |>
                            dplyr::rename(
                                value_1 = .data$value_vector_kernel,
                                response_1 = .data$response
                            ) |>
                            dplyr::left_join(
                                df,
                                by = c("trial_2" = trial_id)
                            ) |>
                            dplyr::rename(
                                value_2 = .data$value_vector_kernel,
                                response_2 = .data$response
                            ) |>
                            dplyr::mutate(
                                rms_distance = abs(.data$value_1 - .data$value_2),
                                agree = as.integer(.data$response_1 == .data$response_2)
                            )
                    }
                )
            ) |>
            dplyr::select(-.data$data) |>
            tidyr::unnest(.data$pair_data) |>
            dplyr::ungroup()

        # plotting it
        # comb_df |>
        #     ggplot(aes(x = rms_distance, y = agree) ) +
        #     # geom_point() +
        #     geom_smooth() +
        #     facet_wrap(~participant)

        # fitting a polynomial regression and returning the intercept
        # link_function <- "probit"
        # fit <- stats::glm(
        #     agree ~ rms_distance,
        #     data = comb_df,
        #     family = stats::binomial(link = link_function),
        #     )

        # GLM predictions at rms_distance = 0
        # pd <- data.frame(rms_distance = c(0, sort(comb_df$rms_distance) ) )
        # pd <- data.frame(rms_distance = 0)
        # preds <- predict(fit, newdata = pd, type = "response", se.fit = FALSE)
        # preds

        # for logit link function
        # plogis(fit$coefficients[1])

        # for probit link function
        # pnorm(fit$coefficients[1])

        # extracting the estimated intercept (in probability space)
        # that is, prob(agree) when trial distance is minimal
        # estimating prop_agree per participant
        message ("Now fitting a GLM to predict prop_agree...\n")

        if (link_function == "logit") {

            consistency <- comb_df |>
                # dplyr::group_by(.data[[participant_id]]) |>
                dplyr::summarise(
                    prop_agree_intercept = as.numeric(stats::plogis(stats::glm(
                        agree ~ rms_distance,
                        family = stats::binomial(link = link_function),
                        )$coefficients[1]) ),
                    .by = dplyr::all_of(summary_groups)
                    ) |>
                # dplyr::ungroup() |>
                data.frame()

        } else if (link_function == "probit") {

            consistency <- comb_df |>
                # dplyr::group_by(.data[[participant_id]]) |>
                dplyr::summarise(
                    prop_agree_intercept = as.numeric(stats::pnorm(stats::glm(
                        agree ~ rms_distance,
                        family = stats::binomial(link = link_function),
                        )$coefficients[1]) ),
                    .by = dplyr::all_of(summary_groups)
                    ) |>
                # dplyr::ungroup() |>
                data.frame()

        }

    } else if (method %in% c("calibrated_kernel", "calibrated_template", "full_glm") ) {

    trial_df <- make_trial_level_data(data)

    feature_cols <- grep("^feature_", names(trial_df), value = TRUE)

    consistency <- trial_df |>
        # dplyr::group_by(.data[[participant_id]]) |>
        dplyr::group_by(dplyr::across(dplyr::all_of(summary_groups) ) ) |>
        dplyr::group_modify(function (df, key) {

            if (length(unique(df$choice_stim2) ) < 2) {

                return (
                    data.frame(
                        calibrated_prop_agree = NA_real_,
                        calibrated_prop_first = mean(
                            df$choice_stim2 == 0,
                            na.rm = TRUE
                            )
                        )
                    )

            }

            x <- as.matrix(df[, feature_cols, drop = FALSE])

            if (method == "calibrated_kernel") {

                positive_kernel <- colMeans(
                    x[df$choice_stim2 == 1, , drop = FALSE]
                    )

                negative_kernel <- colMeans(
                    x[df$choice_stim2 == 0, , drop = FALSE]
                    )

                kernel <- positive_kernel - negative_kernel

                df$score <- as.numeric(x %*% kernel)

                p_hat <- safe_glm_fit_predict(
                    train_data = df,
                    test_data = df,
                    formula = choice_stim2 ~ score,
                    link_function = link_function
                    )

            } else if (method == "calibrated_template") {

                positive_template <- colMeans(
                    x[df$choice_stim2 == 1, , drop = FALSE]
                    )

                negative_template <- colMeans(
                    x[df$choice_stim2 == 0, , drop = FALSE]
                    )

                distance_positive <- sqrt(rowSums(
                    sweep(x, 2, positive_template, "-")^2
                    ))

                distance_negative <- sqrt(rowSums(
                    sweep(x, 2, negative_template, "-")^2
                    ))

                df$score <- distance_negative - distance_positive

                p_hat <- safe_glm_fit_predict(
                    train_data = df,
                    test_data = df,
                    formula = choice_stim2 ~ score,
                    link_function = link_function
                    )

            } else if (method == "full_glm") {

                formula <- stats::as.formula(
                    paste(
                        "choice_stim2 ~",
                        paste(feature_cols, collapse = " + ")
                        )
                    )

                p_hat <- safe_glm_fit_predict(
                    train_data = df,
                    test_data = df,
                    formula = formula,
                    link_function = link_function
                    )

            }

            data.frame(
                calibrated_prop_agree = expected_agreement_from_prob(p_hat),
                calibrated_prop_first = mean(df$choice_stim2 == 0, na.rm = TRUE)
                )

        }) |>
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
                # stim = cumsum(c(1, diff(.data$response) != 0) ),
                stim = cumsum(c(1, diff(.data[[response_id]]) != 0) ),
                .by = c(.data[[participant_id]], .data[[block_id]], .data[[trial_id]])
                ) |>
            dplyr::select(
                .data[[participant_id]], .data[[trial_id]],
                .data[[block_id]], .data[[response_id]], .data$stim
                ) |>
            dplyr::group_by(.data[[participant_id]]) |>
            dplyr::filter(.data[[response_id]] == 1) |>
            dplyr::mutate(
                double_pass_prop_first = dplyr::if_else(
                    .data$stim == 1 & .data[[response_id]] == 1, 1, 0
                    )
                ) |>
            dplyr::summarise(
                double_pass_prop_first = mean(.data$double_pass_prop_first)
                ) |>
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
            dplyr::select(
                .data[[participant_id]], .data[[block_id]],
                .data[[trial_id]], .data$resp
                ) |>
            # removing duplicated rows
            dplyr::distinct() |>
            # reshaping the trial variable
            dplyr::mutate(
                trial = 1:dplyr::n(),
                .by = c(.data[[participant_id]], .data[[block_id]])
                ) |>
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
            by = participant_id
            )

        consistency <- dplyr::left_join(
            consistency,
            double_pass_prop_first,
            by = participant_id
            )

    }

    # returning the consistency metrics
    return (consistency)

}
