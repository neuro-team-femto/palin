#' Computing the kernel from 2AFC data
#'
#' Computing the kernel from 2AFC data. It should be noted that when using the
#' "glm" method, the group-level (average) kernel is simply averaged from the
#' individual-level kernels. To obtain **shrunk** (i.e., adjusted) group-level
#' kernels, one needs to specify the "glmm" method.
#'
#' @param data Dataframe, with reverse correlation data (in long format).
#' @param participant_id Numeric/Character/Factor, column in data specifying the participant ID.
#' @param block_id Numeric, column in data specifying the block ID.
#' @param trial_id Numeric, column in data specifying the trial ID.
#' @param feature_id Numeric/Factor, column in data specifying the feature.
#' @param value_id Numeric, column in data specifying the feature value.
#' @param response_id Numeric, column in data specifying the response.
#' @param method Character, the kernel computation method, either "difference", "GLM", or "GLMM".
#' @param max_feature_levels Numeric, maximum number of feature levels in the GLMM (keep it low to keep GLMM fitting feasible).
#' @param double_pass Logical, indicating whether the last block was repeated.
#'
#' @return The original data augmented with the kernel.
#'
#' @importFrom rlang .data
#'
#' @examples
#' \dontrun{
#' # importing the self-voice data
#' data(self_voice)
#' head(self_voice)
#'
#' # computing the kernel per participant
#' computing_kernel(data = self_voice, method = "difference") |> head()
#' computing_kernel(data = self_voice, method = "GLM") |> head()
#'
#' # plotting it
#' computing_kernel(data = self_voice, method = "difference") |> plot()
#' computing_kernel(data = self_voice, method = "GLM") |> plot()
#'
#' # computing the kernel using a Bayesian multilevel GLM
#' # note that this option is much slower...
#' kernel_glmm <- computing_kernel(data = self_voice, method = "GLMM")
#' }
#'
#' @author Ladislas Nalborczyk \email{ladislas.nalborczyk@@gmail.com}.
#'
#' @export

computing_kernel <- function (
        data,
        participant_id = "participant", block_id = "block", trial_id = "trial",
        feature_id = "feature", value_id = "value", response_id = "response",
        method = c("difference", "GLM", "GLMM"),
        max_feature_levels = 5,
        double_pass = TRUE
        ) {

    # ensuring that the method is one of the above
    method <- match.arg(method)

    # some tests for variable types
    stopifnot("data must be a dataframe..." = is.data.frame(data) )
    stopifnot("method must be a character..." = is.character(method) )
    stopifnot("double_pass must be a logical..." = is.logical(double_pass) )

    # checking required column names
    required_columns <- c(participant_id, block_id, trial_id, feature_id, value_id, response_id)
    assertthat::assert_that(
        all(required_columns %in% colnames(data) ),
        msg = paste(
            "Missing columns:",
            paste(setdiff(required_columns, colnames(data) ), collapse = ", ")
            )
        )

    # if double-pass was used, remove the last (repeated) block
    if (double_pass) {

        nblocks <- dplyr::n_distinct(data[[block_id]])
        if (nblocks > 1) data <- data |> dplyr::filter(.data[[block_id]] != max(.data[[block_id]]) )

    }

    if (method == "difference") {

        # first-order kernel computed for each subject as
        # mean(filter gains of voices classified as mine) - mean(filter gains of voices classified as not-mine)
        kernels <- data |>
            # grouping by participant, feature, and response
            dplyr::group_by(.data[[participant_id]], .data[[feature_id]], .data[[response_id]]) |>
            # computing the average gain
            dplyr::summarise(mean_value = mean(.data[[value_id]]) ) |>
            dplyr::ungroup() |>
            # reshaping the response column in 0/1 (if needed)
            dplyr::mutate(response = as.numeric(as.factor(.data[[response_id]]) )-1) |>
            dplyr::select(-.data[[response_id]]) |>
            # tidyr::pivot_wider(names_from = .data[[response_id]], values_from = .data$mean_value) |>
            tidyr::pivot_wider(names_from = .data$response, values_from = .data$mean_value) |>
            # renaming the columns
            dplyr::rename(negative = .data$`0`, positive = .data$`1`) |>
            # computing the kernel
            # dplyr::filter(.data$f != 0) |>
            dplyr::mutate(kernel_gain = .data$positive - .data$negative) |>
            # kernels are then normalized for each participant by dividing them
            # by the square root of the sum of their squared values
            dplyr::group_by(.data[[participant_id]]) |>
            # as in the code of Ponsot et al. smile paper
            # dplyr::mutate(norm_gain1 = .data$kernel_gain / sqrt(mean(.data$kernel_gain^2) ) ) |>
            # as in the Ponsot et al. PNAS paper and Goupil et al. NatureCom paper
            # dplyr::mutate(norm_gain2 = .data$kernel_gain / sum(abs(.data$kernel_gain) ) ) |>
            # or just the RMSE normalised kernel gain
            dplyr::mutate(norm_kernel_gain = .data$kernel_gain / sqrt(mean(.data$kernel_gain^2) ) ) |>
            dplyr::ungroup()

        # setting the class of the resulting object
        class(kernels) <- c("kernel", "data.frame")

    } else if (method == "GLM") {

        # constructing the formula dynamically
        formula_str <- paste(response_id, "~ 1 +", value_id, "*", feature_id)
        formula_obj <- stats::as.formula(formula_str)

        # fitting a generalised (binomial) linear model
        model <- stats::glm(
            formula = formula_obj,
            family = stats::binomial(),
            data = data |> dplyr::mutate(
                dplyr::across(tidyselect::all_of(feature_id), as.factor)
                )
            )

        # retrieving the coefficients (slopes)
        coefficients <- stats::coef(model)

        # retrieving the confidence intervals
        conf_intervals <- stats::confint(model)

        # identifying the reference level
        reference_level <- levels(as.factor(data[[feature_id]]) )[1]
        reference_slope <- coefficients[value_id]
        reference_ci <- conf_intervals[value_id, ]

        # extracting the interaction terms
        interaction_terms <- grep("value:feature", names(coefficients), value = TRUE)
        interaction_coefs <- coefficients[interaction_terms]
        interaction_cis <- conf_intervals[interaction_terms, , drop = FALSE]

        # computing the slopes for each feature level
        feature_levels <- c(reference_level, gsub("value:feature", "", interaction_terms) )
        slopes <- as.numeric(c(reference_slope, reference_slope + interaction_coefs) )
        ci_lower <- as.numeric(c(reference_ci[1], reference_ci[1] + interaction_cis[, 1]) )
        ci_upper <- as.numeric(c(reference_ci[2], reference_ci[2] + interaction_cis[, 2]) )

        # creating a dataframe with results
        kernels <- data.frame(
            feature = feature_levels,
            kernel = slopes,
            lower = ci_lower,
            upper = ci_upper
            )

        # setting the class of the resulting object
        class(kernels) <- c("glm_kernel", "data.frame")

    # } else if (method == "glmm") {
    #
    #     # reshaping the data
    #     data2 <- data |>
    #         # removing the double-pass block (if needed)
    #         dplyr::filter(if (double_pass) .data[[block_id]] < max(.data[[block_id]]) else TRUE) |>
    #         dplyr::mutate(feature = as.factor(.data[[feature_id]]) ) |>
    #         dplyr::mutate(participant = as.factor(.data[[participant_id]]) ) |>
    #         # converting to data table
    #         data.table::as.data.table()
    #
    #     # fitting a multilevel generalised (binomial) linear model per participant
    #     # this does not work for the moment...
    #     # model <- lme4::glmer(
    #     #     formula = resp ~ 1 + eq * f + (1 + eq | participant) + (1 + eq | participant:f),
    #     #     family = stats::binomial(),
    #     #     data = data2
    #     #     )
    #     # model2 <- lme4::glmer(
    #     #     formula = resp ~ 1 + eq * f + (1 + eq * f | participant),
    #     #     family = stats::binomial(),
    #     #     data = data |> dplyr::mutate(f = as.factor(.data$f) )
    #     #     )
    #
    #     # constructing the formula dynamically
    #     formula_str <- paste(
    #         response_id, "~ 1 +", value_id, "*", feature_id,
    #         "+ s(", participant_id, ", bs = 're')",
    #         "+ s(", value_id, ",", feature_id, ",", participant_id, ", bs = 're')"
    #         )
    #     formula_obj <- as.formula(formula_str)
    #
    #     # using a GAM instead
    #     # see https://m-clark.github.io/posts/2019-10-20-big-mixed-models/
    #     gamm <- mgcv::bam(
    #         formula = formula_obj,
    #             # old formula (working)
    #             # resp ~ 1 + eq * f +
    #             # s(participant, bs = "re") +
    #             # s(eq, f, participant, bs = "re"),
    #             # old formula (not working)
    #             # s(participant, by = eq, bs = "re"),
    #             # s(participant, by = f, bs = "re") +
    #             # s(participant, by = interaction(eq, f), bs = "re"),
    #         family = stats::binomial(link = "logit"),
    #         data = data2,
    #         discrete = TRUE,
    #         # https://stackoverflow.com/questions/78590558/how-can-i-enable-openmp-in-mac-os-for-mgcv-r-package
    #         nthreads = parallel::detectCores() - 1,
    #         method = "fREML"
    #         )
    #
    #     ############################
    #     # retrieving fixed effects #
    #     ############################
    #
    #     # retrieving model coefficients
    #     coefficients <- stats::coef(gamm)
    #
    #     # main effect of value
    #     value_coef <- coefficients[value_id]
    #
    #     # interaction terms
    #     value_feauture_interaction <- coefficients[grepl("^value:feature", names(coefficients) )]
    #
    #     # creating a table of slopes for eq at each f
    #     group_kernel <- data.frame(
    #         feature = sub("value:feature", "", names(value_feauture_interaction) ),
    #         kernel_gain = value_coef + value_feauture_interaction
    #         ) |>
    #         dplyr::mutate(participant = "group") |>
    #         dplyr::relocate(.data[[participant_id]], 1)
    #
    #     # removing rownames
    #     rownames(group_kernel) <- NULL
    #
    #     # generating a grid of possible values for predictions
    #     # new_data <- expand.grid(
    #     #     eq = unique(data2$eq),
    #     #     f = unique(data2$f),
    #     #     participant = NA
    #     #     )
    #
    #     # retrieving predictions with SE
    #     # new_data$predicted <- predict(
    #     #     gamm, newdata = new_data, type = "link",
    #     #     exclude = "s(participant)",
    #     #     se.fit = TRUE
    #     #     )
    #
    #     #############################
    #     # retrieving random effects #
    #     #############################
    #
    #     # creating dataset with small variations in eq to estimate slopes
    #     new_data <- tidyr::crossing(
    #         participant = unique(data2$participant),
    #         f = unique(data2$f)
    #         ) |>
    #         dplyr::mutate(
    #             # small perturbation below mean eq
    #             eq_low = mean(data2$eq) - 0.01,
    #             # small perturbation above mean eq
    #             eq_high = mean(data2$eq) + 0.01
    #             )
    #
    #     # predicting values at slightly different eq levels
    #     new_data$pred_low <- stats::predict(
    #         gamm, newdata = dplyr::mutate(new_data, eq = .data$eq_low),
    #         type = "link"
    #         )
    #     new_data$pred_high <- stats::predict(
    #         gamm, newdata = dplyr::mutate(new_data, eq = .data$eq_high),
    #         type = "link"
    #         )
    #
    #     # computing approximate slope (kernel) as finite difference
    #     kernel <- new_data |>
    #         dplyr::mutate(
    #             kernel_gain = (.data$pred_high - .data$pred_low) /
    #                 (.data$eq_high - .data$eq_low)
    #             ) |>
    #         dplyr::select(.data$participant, .data$f, .data$kernel_gain)
    #
    #     # appending fixed and random effects
    #     kernel <- dplyr::bind_rows(kernel, group_kernel)
    #
    #     # setting the class of the resulting object
    #     class(kernel) <- c("glmm_kernel", "data.frame")

    } else if (method == "GLMM") {

        # retrieving the unique feature levels
        unique_feature_levels <- unique(data[[feature_id]])

        if (length(unique_feature_levels <- unique(data[[feature_id]]) ) > max_feature_levels) {

            # subsetting the feature levels
            unique_feature_levels <- unique_feature_levels[1:max_feature_levels]

            # warning the use that we did subset the data
            cat(
            "Beware that we only retained the first", max_feature_levels, "feature levels to maintain a reasonable fitting time...\n"
            )

            # reshaping the data
            data2 <- data |>
                # removing the double-pass block (if needed)
                dplyr::filter(if (double_pass) .data[[block_id]] < max(.data[[block_id]]) else TRUE) |>
                # reshaping the original variables
                dplyr::mutate(feature = as.factor(.data[[feature_id]]) ) |>
                dplyr::mutate(participant = .data[[participant_id]]) |>
                dplyr::mutate(response = .data[[response_id]]) |>
                dplyr::mutate(value = .data[[value_id]]) |>
                # removing some features to keep the fitting reasonable
                dplyr::filter(.data$feature %in% unique_feature_levels) |>
                # removing potential NAs
                stats::na.omit()

        } else {

            # reshaping the data
            data2 <- data |>
                # removing the double-pass block (if needed)
                dplyr::filter(if (double_pass) .data[[block_id]] < max(.data[[block_id]]) else TRUE) |>
                # reshaping the original variables
                dplyr::mutate(feature = as.factor(.data[[feature_id]]) ) |>
                dplyr::mutate(participant = .data[[participant_id]]) |>
                dplyr::mutate(response = .data[[response_id]]) |>
                dplyr::mutate(value = .data[[value_id]]) |>
                # removing potential NAs
                stats::na.omit()

        }

        # Bayesian GLMM approach with varying intercept and slope per participant
        glmm <- brms::brm(
            # formula = response ~ 1 + value * feature + (1 + value * feature | participant),
            # formula = response ~ 1 + value * feature + (1 + value * feature || participant),
            formula = response ~ 0 + value * feature + (0 + value * feature || participant),
            family = brms::bernoulli(),
            data = data2,
            # prior = c(
            #     brms::set_prior(prior = "normal(0, 1)", class = "b"),
            #     brms::set_prior(prior = "exponential(1)", class = "sd")
            #     ),
            warmup = 1000,
            iter = 2000,
            chains = 4,
            cores = 4
            )

        # retrieving the group-level (fixed/constant) effects
        fixed_effects <- brms::fixef(glmm)
        group_kernel <- fixed_effects |>
            as.data.frame() |>
            tibble::rownames_to_column(var = "variable") |>
            dplyr::filter(grepl("value", .data$variable) ) |>
            dplyr::mutate(participant = "group", .before = .data$variable) |>
            dplyr::mutate(variable =  sub("\\..*", "", .data$variable) ) |>
            dplyr::mutate(variable =  sub(".*:", "", .data$variable) )

        # reshaping to provide the group-level kernel
        group_kernel <- data.frame(
            feature = group_kernel$variable,
            kernel = group_kernel$Estimate,
            lower = group_kernel$Q2.5,
            upper = group_kernel$Q97.5
            ) |>
            dplyr::mutate(participant = "group", .before = .data$feature)

        # retrieving the individual-level (random/varying) effects
        participant_slopes <- stats::coef(glmm)$participant
        dimnames(participant_slopes)[[3]] <- sub("\\..*", "", dimnames(participant_slopes)[[3]])
        participant_slopes <- data.frame(participant_slopes) |>
            tibble::rownames_to_column(var = "participant")

        # keeping only the columns that contains "value"
        participant_slopes <- participant_slopes |> dplyr::select(.data$participant, tidyselect::contains("value") )

        # reshaping into long format
        kernels <- participant_slopes |>
            tidyr::pivot_longer(
                cols = -.data$participant,
                names_to = c(".value", "feature"),
                names_sep = "(?<=.)\\.(?=[^\\.]+$)" # regex to split on the last period
                ) |>
            dplyr::mutate(
                kernel = dplyr::coalesce(.data$Estimate.value, .data$Estimate),
                lower = dplyr::coalesce(.data$Q2.5.value, .data$Q2.5),
                upper = dplyr::coalesce(.data$Q97.5.value, .data$Q97.5)
                ) |>
            dplyr::select(.data$participant, .data$feature, .data$kernel, .data$lower, .data$upper)

        # appending fixed and random effects
        kernels <- dplyr::bind_rows(kernels, group_kernel) |>
            dplyr::mutate(feature = unique_feature_levels, .by = .data$participant)

        # setting the class of the resulting object
        class(kernels) <- c("glmm_kernel", "data.frame")

    }

    # returning the kernels
    return (kernels)

}

#' @export

plot.kernel <- function (
        x, normalisation_method = c("kernel_gain", "norm_kernel_gain"),
        log = TRUE,
        ...
        ) {

    # ensuring that the normalisation method is one of the above
    normalisation_method <- match.arg(normalisation_method)

    # plotting the filters
    x |>
        # avoiding log(0)
        # dplyr::filter(.data$feature > 0) |>
        ggplot2::ggplot(
            ggplot2::aes(
                x = .data$feature,
                y = .data$kernel_gain,
                colour = .data$participant,
                fill = .data$participant
                )
            ) +
        ggplot2::geom_hline(yintercept = 0, linetype = 2) +
        # plotting each participant's filter
        ggplot2::geom_line(size = 0.5, alpha = 0.3, show.legend = FALSE) +
        # plotting the average filter
        ggplot2::stat_summary(
            ggplot2::aes(x = .data$feature, y = .data[[normalisation_method]]),
            geom = "line",
            fun = mean,
            size = 1,
            inherit.aes = FALSE
            ) +
        # aesthetics
        # ggplot2::theme_bw(base_family = "Open Sans", base_size = 12) +
        ggplot2::theme_bw() +
        ggplot2::scale_colour_manual(
            values = MetBrewer::met.brewer(
                name = "Johnson",
                n = dplyr::n_distinct(x$participant)
                )
            ) +
        ggplot2::scale_fill_manual(
            values = MetBrewer::met.brewer(
                name = "Johnson",
                n = dplyr::n_distinct(x$participant)
                )
            ) +
        {if (normalisation_method == "kernel_gain") ggplot2::labs(
            x = "Feature",
            y = "Kernel amplitude (a.u.)"
            )} +
        {if (normalisation_method != "kernel_gain") ggplot2::labs(
            x = "Feature",
            y = "Normalised kernel amplitude (a.u.)"
            )} +
        {if (log) ggplot2::scale_x_log10(
            breaks = scales::breaks_log(n = 5, base = 10),
            labels = scales::label_log(base = 10)
            )} +
        {if (log) ggplot2::annotation_logticks(sides = "b")}

}

#' @export

plot.glm_kernel <- function (x, log = TRUE, ...) {

    x |>
        # avoiding log(0)
        # dplyr::filter(.data$feature > 0) |>
        ggplot2::ggplot(
            ggplot2::aes(
                x = as.numeric(.data$feature),
                y = .data$kernel
                )
            ) +
        ggplot2::geom_hline(yintercept = 0, linetype = 2) +
        # plotting the group-level average kernel
        ggplot2::geom_ribbon(
            ggplot2::aes(ymin = .data$lower, ymax = .data$upper),
            alpha = 0.2
            ) +
        ggplot2::geom_line(linewidth = 1) +
        ggplot2::labs(
            x = "Feature",
            y = "Kernel amplitude (a.u.)"
            ) +
        # aesthetics
        ggplot2::theme_bw() +
        ggplot2::scale_colour_manual(
            values = MetBrewer::met.brewer(
                name = "Johnson",
                n = dplyr::n_distinct(x$participant)
                )
            ) +
        ggplot2::theme(legend.position = "none") +
        {if (log) ggplot2::scale_x_log10(
            breaks = scales::breaks_log(n = 5, base = 10),
            labels = scales::label_log(base = 10)
            )} +
        {if (log) ggplot2::annotation_logticks(sides = "b")}

}

#' @export

plot.glmm_kernel <- function (x, log = TRUE, ...) {

    x |>
        # avoiding log(0)
        # dplyr::filter(.data$feature > 0) |>
        ggplot2::ggplot(
            ggplot2::aes(
                x = .data$feature,
                y = .data$kernel,
                group = .data$participant,
                color = .data$participant
                )
            ) +
        ggplot2::geom_hline(yintercept = 0, linetype = 2) +
        # plotting individual kernels
        ggplot2::geom_line(
            data = x |> dplyr::filter(.data$participant != "group"),
            linewidth = 0.5, alpha = 0.5
            ) +
        # plotting the group-level average kernel
        ggplot2::geom_ribbon(
            data = x |> dplyr::filter(.data$participant == "group"),
            ggplot2::aes(ymin = .data$lower, ymax = .data$upper, colour = NULL),
            fill = "gray",
            alpha = 0.5
            ) +
        ggplot2::geom_line(
            data = x |> dplyr::filter(.data$participant == "group"),
            colour = "black",
            linewidth = 1
            ) +
        ggplot2::labs(
            x = "Feature",
            y = "Kernel amplitude (a.u.)"
            ) +
        # aesthetics
        ggplot2::theme_bw() +
        ggplot2::scale_colour_manual(
            values = MetBrewer::met.brewer(
                name = "Johnson",
                n = dplyr::n_distinct(x$participant)
                )
            ) +
        ggplot2::theme(legend.position = "none") +
        {if (log) ggplot2::scale_x_log10(
            breaks = scales::breaks_log(n = 5, base = 10),
            labels = scales::label_log(base = 10)
            )} +
        {if (log) ggplot2::annotation_logticks(sides = "b")}

}
